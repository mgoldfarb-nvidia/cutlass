# Copyright (c) 2026 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""JAX ``cutlass_call`` runner for grouped mixed-input GEMM examples."""

from dataclasses import dataclass
import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import cutlass
import cutlass.cute.testing as testing
import cutlass.utils as utils
import cutlass.utils.mixed_input_helpers as mixed_input_utils
from cutlass.jax import TensorSpec, cutlass_call, cutlass_to_jax_dtype
from cutlass.jax.testing import (
    gemm_a_major,
    gemm_a_mode,
    gemm_a_shape,
    gemm_b_major,
    gemm_b_mode,
    gemm_b_shape,
    gemm_c_major,
    gemm_c_mode,
    gemm_c_shape,
)


@dataclass
class _JaxWorkspace:
    """Input and output arrays for one grouped GEMM invocation."""

    a: jax.Array
    a_ref: jax.Array
    a_scale: jax.Array
    b: jax.Array
    cumsum: jax.Array
    c: jax.Array
    cumsum_np: np.ndarray


def _logical_from_physical(
    tensor: jax.Array, physical_order: str, logical_order: str
) -> jax.Array:
    axes = tuple(physical_order.index(mode) for mode in logical_order)
    return jnp.transpose(tensor, axes)


def _physical_from_logical_np(
    tensor: np.ndarray, logical_order: str, physical_order: str
) -> np.ndarray:
    axes = tuple(logical_order.index(mode) for mode in physical_order)
    return np.transpose(tensor, axes)


def _physical_from_logical(
    tensor: jax.Array, logical_order: str, physical_order: str
) -> jax.Array:
    axes = tuple(logical_order.index(mode) for mode in physical_order)
    return jnp.transpose(tensor, axes)


def _shuffle_a_for_kernel(
    a: jax.Array,
    *,
    m: int,
    k: int,
    group_count: int,
    a_major: str,
) -> jax.Array:
    """Match the native Int4/BF16 shuffle-A host transform."""
    if a_major != "k" or k % 8 != 0:
        return a

    perm = jnp.asarray((0, 2, 1, 3, 4, 6, 5, 7), dtype=jnp.int32)
    a_logical = _logical_from_physical(a, gemm_a_major(a_major), "mkl")
    a_logical = a_logical.reshape(m, k // 8, 8, group_count)
    a_logical = jnp.take(a_logical, perm, axis=2)
    a_logical = a_logical.reshape(m, k, group_count)
    return _physical_from_logical(a_logical, "mkl", gemm_a_major(a_major))


def _random_tensor(
    key: jax.Array,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
    minval: float,
    maxval: float,
) -> jax.Array:
    tensor = jax.random.uniform(
        key, shape, dtype=jnp.float32, minval=minval, maxval=maxval
    )
    return tensor.astype(dtype)


def _random_int4_tensor(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    tensor = jax.random.randint(key, shape, minval=-7, maxval=7, dtype=jnp.int32)
    return tensor.astype(jnp.int4)


def _random_integer_tensor(
    key: jax.Array,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
    minval: int,
    maxval: int,
) -> jax.Array:
    tensor = jax.random.randint(
        key, shape, minval=minval, maxval=maxval, dtype=jnp.int32
    )
    return tensor.astype(dtype)


def _random_cumsum(
    group_count: int,
    total_n: int,
    alignment: int,
    *,
    uniform_group_sizes: bool,
    seed: int,
) -> np.ndarray:
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    if total_n % alignment != 0:
        raise ValueError(
            f"total_n={total_n} must be divisible by alignment={alignment}"
        )

    if uniform_group_sizes:
        if total_n % group_count != 0:
            raise ValueError(
                f"Uniform groups require total_n={total_n} divisible by "
                f"group_count={group_count}"
            )
        group_counts = np.full(group_count, total_n // group_count, dtype=np.int32)
    else:
        rng = np.random.default_rng(seed)
        assignments = rng.integers(0, group_count, size=total_n // alignment)
        group_counts = np.bincount(assignments, minlength=group_count).astype(np.int32)
        group_counts *= alignment

    cumsum = np.concatenate(
        [np.array([0], dtype=np.int32), np.cumsum(group_counts, dtype=np.int32)]
    )
    return cumsum


def _shape_num_elements(shape: tuple[int, ...]) -> int:
    num_elements = 1
    for extent in shape:
        num_elements *= extent
    return num_elements


def _packed_size_bytes(shape: tuple[int, ...], dtype: type[cutlass.Numeric]) -> int:
    return (_shape_num_elements(shape) * dtype.width + 7) // 8


def _workspace_size_bytes(
    a_shape: tuple[int, ...],
    a_scale_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    c_shape: tuple[int, ...],
    cumsum_shape: tuple[int, ...],
    *,
    a_dtype: type[cutlass.Numeric],
    b_dtype: type[cutlass.Numeric],
    c_dtype: type[cutlass.Numeric],
) -> int:
    return (
        _packed_size_bytes(a_shape, a_dtype)
        + _packed_size_bytes(a_scale_shape, b_dtype)
        + _packed_size_bytes(b_shape, b_dtype)
        + _packed_size_bytes(c_shape, c_dtype)
        + _shape_num_elements(cumsum_shape) * np.dtype(np.int32).itemsize
    )


def _comparison_rtol(c_dtype: type[cutlass.Numeric]) -> float:
    if c_dtype == cutlass.BFloat16:
        return 2**-7
    if c_dtype == cutlass.Float16:
        return 2**-10
    return 1e-5


def _create_workspace(
    *,
    m: int,
    k: int,
    n: int,
    group_count: int,
    a_shape: tuple[int, ...],
    a_scale_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    c_shape: tuple[int, ...],
    b_dtype: type[cutlass.Numeric],
    c_dtype: type[cutlass.Numeric],
    uniform_group_sizes: bool,
    shuffle_a: bool,
    keep_reference_a: bool,
    a_major: str,
    seed: int,
) -> _JaxWorkspace:
    b_jax_dtype = cutlass_to_jax_dtype(b_dtype)
    c_jax_dtype = cutlass_to_jax_dtype(c_dtype)
    key_a, key_scale, key_b = jax.random.split(jax.random.key(seed), 3)
    cumsum_np = _random_cumsum(
        group_count,
        n * group_count,
        16 * 8 // b_dtype.width,
        uniform_group_sizes=uniform_group_sizes,
        seed=seed,
    )
    a_ref = _random_int4_tensor(key_a, a_shape)
    a = (
        _shuffle_a_for_kernel(
            a_ref,
            m=m,
            k=k,
            group_count=group_count,
            a_major=a_major,
        )
        if shuffle_a
        else a_ref
    )
    return _JaxWorkspace(
        a=a,
        a_ref=a_ref if keep_reference_a else a,
        a_scale=_random_integer_tensor(key_scale, a_scale_shape, b_jax_dtype, -3, 3),
        b=_random_tensor(key_b, b_shape, b_jax_dtype, -10.0, 10.0),
        cumsum=jnp.asarray(cumsum_np, dtype=jnp.int32),
        c=jnp.zeros(c_shape, dtype=c_jax_dtype),
        cumsum_np=cumsum_np,
    )


def _reference(
    a: jax.Array,
    a_scale: jax.Array,
    b: jax.Array,
    cumsum: np.ndarray,
    *,
    m: int,
    n: int,
    k: int,
    group_count: int,
    scale_granularity_m: int,
    scale_granularity_k: int,
    a_major: str,
    b_major: str,
    c_major: str,
    preferred_element_type: jnp.dtype,
    scale_after_accumulation: bool,
) -> np.ndarray:
    a_logical = _logical_from_physical(a, gemm_a_major(a_major), "mkl").astype(
        jnp.float32
    )
    a_scale_logical = _logical_from_physical(a_scale, gemm_a_major("m"), "mkl").astype(
        jnp.float32
    )
    b_logical = _logical_from_physical(b, gemm_b_major(b_major), "nkl").astype(
        jnp.float32
    )

    del group_count, scale_after_accumulation
    a_scale_logical = jnp.repeat(a_scale_logical, scale_granularity_m, axis=0)[:m, :, :]
    lhs = jnp.squeeze(b_logical, axis=2)
    group_sizes = jnp.asarray(cumsum[1:] - cumsum[:-1], dtype=jnp.int32)
    a_scale_logical = jnp.repeat(a_scale_logical, scale_granularity_k, axis=1)[:, :k, :]
    rhs = jnp.transpose(a_logical * a_scale_logical, (2, 1, 0))
    ref = jax.lax.ragged_dot(
        lhs,
        rhs,
        group_sizes,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    ref_logical = jnp.expand_dims(jnp.transpose(ref, (1, 0)), axis=2)
    ref_logical = ref_logical.astype(preferred_element_type)
    return _physical_from_logical_np(
        np.asarray(ref_logical), "mnl", gemm_c_major(c_major)
    )


def run_grouped_mixed_input_gemm_jax(
    kernel_cls: type,
    mnkl: tuple[int, int, int, int],
    scale_granularity_m: int,
    scale_granularity_k: int,
    a_dtype: type[cutlass.Numeric],
    b_dtype: type[cutlass.Numeric],
    c_dtype: type[cutlass.Numeric],
    acc_dtype: type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    mma_tiler_mnk: tuple[int, int, int],
    cluster_shape_mn: tuple[int, int],
    use_2cta_instrs: bool,
    tolerance: float,
    *,
    skip_ref_check: bool = False,
    uniform_group_sizes: bool = False,
    warmup_iterations: int = 0,
    iterations: int = 1,
    use_cold_l2: bool = False,
    scale_after_accumulation: bool = False,
    seed: int = 2025,
) -> float | None:
    """Run a grouped mixed-input GEMM through JAX ``cutlass_call``."""
    if a_dtype != cutlass.Int4:
        raise ValueError("--jax currently supports Int4 A for grouped mixed-input GEMM")
    if scale_granularity_m <= 0 or scale_granularity_k <= 0:
        raise ValueError("--jax requires convert-scale mode with positive granularity")
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")

    m, n, k, group_count = mnkl
    if m % scale_granularity_m != 0:
        raise ValueError(f"M={m} must be divisible by scale_granularity_m")
    if k % scale_granularity_k != 0:
        raise ValueError(f"K={k} must be divisible by scale_granularity_k")

    if not kernel_cls.can_implement(
        mnkl,
        a_dtype,
        b_dtype,
        c_dtype,
        a_major,
        b_major,
        c_major,
        scale_granularity_m,
        scale_granularity_k,
        mma_tiler_mnk,
        cluster_shape_mn,
        use_2cta_instrs,
    ):
        raise ValueError("GEMM configuration not supported")

    shuffle_a = mixed_input_utils.is_shuffle_a(
        a_major, k, a_dtype, b_dtype, scale_granularity_k
    )
    kernel = kernel_cls(
        scale_granularity_m,
        scale_granularity_k,
        acc_dtype,
        use_2cta_instrs,
        mma_tiler_mnk,
        cluster_shape_mn,
        group_count,
        shuffle_a,
    )

    c_jax_dtype = cutlass_to_jax_dtype(c_dtype)
    a_shape = gemm_a_shape(group_count, m, k, a_major)
    a_scale_shape = gemm_a_shape(
        group_count,
        m // scale_granularity_m,
        k // scale_granularity_k,
        "m",
    )
    b_shape = gemm_b_shape(1, n * group_count, k, b_major)
    c_shape = gemm_c_shape(1, m, n * group_count, c_major)
    cumsum_shape = (group_count + 1,)

    workspace_count = 1
    if use_cold_l2 and iterations > 0:
        one_workspace_bytes = _workspace_size_bytes(
            a_shape,
            a_scale_shape,
            b_shape,
            c_shape,
            cumsum_shape,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            c_dtype=c_dtype,
        )
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    workspaces = [
        _create_workspace(
            m=m,
            k=k,
            n=n,
            group_count=group_count,
            a_shape=a_shape,
            a_scale_shape=a_scale_shape,
            b_shape=b_shape,
            c_shape=c_shape,
            b_dtype=b_dtype,
            c_dtype=c_dtype,
            uniform_group_sizes=uniform_group_sizes,
            shuffle_a=shuffle_a,
            keep_reference_a=not skip_ref_check,
            a_major=a_major,
            seed=seed + workspace_idx,
        )
        for workspace_idx in range(workspace_count)
    ]

    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1],
    )

    def launch(stream, a, a_scale, b, cumsum, c, *, max_active_clusters):
        kernel(a, a_scale, b, cumsum, c, max_active_clusters, stream)

    a_divisibility = mixed_input_utils.get_divisibility(m if a_major == "m" else k)
    a_scale_divisibility = mixed_input_utils.get_divisibility(m // scale_granularity_m)
    b_divisibility = mixed_input_utils.get_divisibility(n if b_major == "n" else k)
    c_divisibility = mixed_input_utils.get_divisibility(m if c_major == "m" else n)
    a_spec = TensorSpec(mode=gemm_a_mode(a_major), divisibility=a_divisibility)
    a_scale_spec = TensorSpec(
        mode=gemm_a_mode("m"),
        divisibility=a_scale_divisibility,
    )
    b_spec = TensorSpec(mode=gemm_b_mode(b_major), divisibility=b_divisibility)
    c_spec = TensorSpec(mode=gemm_c_mode(c_major), divisibility=c_divisibility)

    compiled_call = cutlass_call(
        launch,
        output_shape_dtype=workspaces[0].c,
        input_output_aliases={4: 0},
        input_spec=(
            a_spec,
            a_scale_spec,
            b_spec,
            None,  # Default tensor layout.
            c_spec,
        ),
        output_spec=(c_spec,),
        max_active_clusters=max_active_clusters,
    )
    compiled_call = jax.jit(compiled_call, donate_argnums=[4])

    def run_once(workspace: _JaxWorkspace) -> jax.Array:
        workspace.c = compiled_call(
            workspace.a, workspace.a_scale, workspace.b, workspace.cumsum, workspace.c
        )
        return workspace.c

    first_workspace = workspaces[0]
    c = run_once(first_workspace)

    if not skip_ref_check:
        ref = _reference(
            first_workspace.a_ref,
            first_workspace.a_scale,
            first_workspace.b,
            first_workspace.cumsum_np,
            m=m,
            n=n,
            k=k,
            group_count=group_count,
            scale_granularity_m=scale_granularity_m,
            scale_granularity_k=scale_granularity_k,
            a_major=a_major,
            b_major=b_major,
            c_major=c_major,
            preferred_element_type=c_jax_dtype,
            scale_after_accumulation=scale_after_accumulation,
        )
        np.testing.assert_allclose(
            np.asarray(c).astype(np.float32),
            ref.astype(np.float32),
            atol=tolerance,
            rtol=_comparison_rtol(c_dtype),
        )

    if iterations == 0:
        return None

    workspace_idx = 0
    for _ in range(warmup_iterations):
        x = run_once(workspaces[workspace_idx])
        workspace_idx = (workspace_idx + 1) % workspace_count
    x.block_until_ready()

    start_time = time.perf_counter()
    for _ in range(iterations):
        x = run_once(workspaces[workspace_idx])
        workspace_idx = (workspace_idx + 1) % workspace_count
    x.block_until_ready()
    end_time = time.perf_counter()

    return (end_time - start_time) * 1e6 / iterations
