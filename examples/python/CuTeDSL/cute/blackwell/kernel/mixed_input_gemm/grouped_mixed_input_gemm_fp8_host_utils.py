# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Host-side helpers for grouped FP8 mixed-input GEMM examples."""

from dataclasses import dataclass
from typing import Optional

import torch

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils.mixed_input_helpers as mixed_input_utils

from blackwell.kernel.mixed_input_gemm.mixed_input_host_utils import (
    create_cumsum_tensor,
)


@dataclass(frozen=True)
class TensorAndRef:
    # A reference tensor used in a reference implementation.
    ref_torch: torch.Tensor
    # The CuTe tensor to pass to/from a kernel.
    cute_tensor: cute.Tensor
    # The CUDA backing storage for cute_tensor, when the caller needs it.
    cute_torch: Optional[torch.Tensor]

    def __str__(self):
        s = f"ref_torch: {self.ref_torch.shape} {self.ref_torch.dtype}\n"
        s += f"cute_tensor: {self.cute_tensor} {self.cute_tensor.element_type}\n"
        if self.cute_torch is None:
            s += "cute_torch: None"
        else:
            s += f"cute_torch: {self.cute_torch.shape} {self.cute_torch.dtype}"
        return s

    def __repr__(self):
        return str(self)


@dataclass(frozen=True)
class GroupedMixedInputTensors:
    a: TensorAndRef
    a_scale: Optional[TensorAndRef]
    b: TensorAndRef
    cumsum: TensorAndRef
    c: TensorAndRef
    b_scale: Optional[TensorAndRef] = None


def make_tensor_and_ref(
    ref_torch: torch.Tensor,
    dtype: type[cutlass.Numeric],
    *,
    is_dynamic_layout: bool,
    assumed_align: Optional[int] = None,
) -> TensorAndRef:
    """Create a CuTe tensor and preserve its reference and CUDA backing storage."""
    cute_tensor, cute_torch = cutlass_torch.cute_tensor_like(
        ref_torch,
        dtype,
        is_dynamic_layout=is_dynamic_layout,
        assumed_align=assumed_align,
    )
    return TensorAndRef(ref_torch, cute_tensor, cute_torch)


def create_cumsum_tensor_and_ref(
    num_groups: int,
    fused_n: int,
    alignment: int,
    uniform_distribution: bool = False,
) -> TensorAndRef:
    """Create a cumsum TensorAndRef for FP8 grouped GEMM helpers."""
    cumsum_tensor, cumsum_torch_cpu = create_cumsum_tensor(
        num_groups, fused_n, alignment, uniform_distribution=uniform_distribution
    )
    return TensorAndRef(cumsum_torch_cpu, cumsum_tensor, None)


def _is_fp8_dtype(dtype: type[cutlass.Numeric]) -> bool:
    return dtype in (cutlass.Float8E4M3FN, cutlass.Float8E5M2)


def build_rowsum_tensor(b_tensor_and_ref: TensorAndRef, mma_K: int) -> TensorAndRef:
    """Allocate fp32 rowsum workspace for precomputed-rowsum FP8 prefill kernels."""
    assert b_tensor_and_ref.cute_torch is not None
    b_gpu_int8 = b_tensor_and_ref.cute_torch
    n_tokens = b_gpu_int8.shape[0]
    k_total = b_gpu_int8.shape[1]
    num_chunks = k_total // mma_K
    assert k_total % mma_K == 0, f"k={k_total} not divisible by mma_K={mma_K}"
    rowsum_buf = torch.empty(
        (n_tokens, num_chunks, 1), dtype=torch.float32, device="cuda"
    )
    rowsum_buf = rowsum_buf.permute(2, 1, 0).contiguous().permute(2, 1, 0).contiguous()
    cute_rowsum_tensor, torch_rowsum_tensor = cutlass_torch.cute_tensor_like(
        rowsum_buf,
        cutlass.Float32,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    return TensorAndRef(rowsum_buf, cute_rowsum_tensor, torch_rowsum_tensor)


def create_simple_sfb_scale_tensor(
    n: int,
    k: int,
    scale_granularity_k: int,
    dtype: type[cutlass.Numeric] = cutlass.BFloat16,
    is_dynamic_layout: bool = True,
    divisibility: int = 16,
) -> TensorAndRef:
    """Create a per-(N_tokens, K chunk) SFB tensor in BF16 or E8M0 format."""
    if scale_granularity_k <= 0:
        raise ValueError("scale_granularity_k must be positive")
    assert k % scale_granularity_k == 0, (
        f"K={k} must be divisible by scale_granularity_k={scale_granularity_k}"
    )
    assert dtype in (cutlass.BFloat16, cutlass.Float8E8M0FNU), (
        f"create_simple_sfb_scale_tensor: unsupported dtype {dtype}; "
        "expected BFloat16 or Float8E8M0FNU"
    )
    num_scales = k // scale_granularity_k
    l = 1

    # Powers of two are exactly representable in both BF16 and E8M0.
    exp = torch.empty((n, num_scales, l), dtype=torch.float32).random_(0, 8)
    sfb_ref_fp32 = torch.pow(2.0, exp)
    sfb_ref_fp32 = sfb_ref_fp32.permute(2, 1, 0).contiguous().permute(2, 1, 0)

    return make_tensor_and_ref(
        sfb_ref_fp32,
        dtype,
        is_dynamic_layout=is_dynamic_layout,
        assumed_align=divisibility,
    )


def create_i4_tensor_and_scale_mxf8(
    l: int,
    m: int,
    k: int,
    is_m_major: bool,
    dtype: type[cutlass.Numeric],
    scale_granularity_k: int,
    is_dynamic_layout: bool = True,
    init_config: tuple = (
        cutlass_torch.TensorInitType.RANDOM,
        cutlass_torch.RandomInitConfig(min_val=-7, max_val=6),
    ),
    divisibility: int = 16,
) -> tuple[TensorAndRef, TensorAndRef]:
    """Create an Int4 tensor and BF16 scale tensor for the FP8 grouped path."""
    if scale_granularity_k <= 0:
        raise ValueError("scale_granularity_k must be positive")
    lb_4b = -8 if dtype == cutlass.Int4 else 0
    up_4b = 7 if dtype == cutlass.Int4 else 15
    if not (
        init_config[0] == cutlass_torch.TensorInitType.RANDOM
        or init_config[0] == cutlass_torch.TensorInitType.SCALAR
    ):
        raise ValueError(
            "Only random and scalar initialization is supported for 4bit data type"
        )

    ref_fp32 = cutlass_torch.matrix(l, m, k, is_m_major, cutlass.Float32, *init_config)
    num_scales = k // scale_granularity_k
    ref = ref_fp32.to(dtype=cutlass_torch.dtype(cutlass.BFloat16)).reshape(
        m, num_scales, scale_granularity_k, l
    )

    a_max = (
        torch.maximum(ref / up_4b, ref / lb_4b)
        if dtype == cutlass.Int4
        else ref / up_4b
    )
    a_scales, _ = torch.max(a_max, dim=2, keepdim=True)
    a_scale_inv = torch.where(a_scales == 0, 0, 1 / a_scales)
    a_quant = ref * a_scale_inv
    a_quant = a_quant.to(dtype=torch.int32).reshape((m, k, l)).to(dtype=torch.float32)

    sfa_ref = (a_scales.random_(1, 3)).reshape((m, num_scales, l))
    sfa_ref = sfa_ref.to(dtype=torch.bfloat16)
    sfa_ref = sfa_ref.permute(2, 1, 0).contiguous().permute(2, 1, 0)

    sfa = make_tensor_and_ref(
        sfa_ref,
        cutlass.BFloat16,
        is_dynamic_layout=is_dynamic_layout,
        assumed_align=divisibility,
    )
    a = make_tensor_and_ref(
        a_quant,
        dtype,
        is_dynamic_layout=is_dynamic_layout,
        assumed_align=divisibility,
    )

    return a, sfa


def create_tensors_for_contiguous_grouped_mixed_input_gemm_fp8(
    m: int,
    g: int,
    k: int,
    n: int,
    a_dtype: type[cutlass.Numeric],
    b_dtype: type[cutlass.Numeric],
    c_dtype: type[cutlass.Numeric],
    scale_granularity_k: int,
    uniform_group_sizes: bool = True,
    is_dynamic_layout: bool = True,
    sfb_dtype: type[cutlass.Numeric] = cutlass.BFloat16,
    sfb_granularity_k: Optional[int] = None,
    multiply_n_by_num_groups: bool = False,
) -> GroupedMixedInputTensors:
    """Create host tensors for the FP8 grouped/MoE GEMM setup.

    This helper intentionally stays separate from the non-FP8 grouped helper:
    ``n`` is total tokens by default, output is token-major, and B has an
    additional SFB scale tensor.
    """
    assert a_dtype == cutlass.Int4, (
        f"MoE expects Int4 for weights (a_dtype), got {a_dtype}"
    )
    assert _is_fp8_dtype(b_dtype), (
        f"MoE expects fp8 for activations (b_dtype), got {b_dtype}"
    )
    total_n = n * g if multiply_n_by_num_groups else n
    if uniform_group_sizes:
        assert total_n % g == 0, (
            f"Uniform groups require total_n={total_n} divisible by g={g}"
        )

    a, a_scale = create_i4_tensor_and_scale_mxf8(
        g,
        m,
        k,
        False,
        a_dtype,
        scale_granularity_k,
        is_dynamic_layout,
    )

    b_torch_cpu = cutlass_torch.matrix(
        1,
        total_n,
        k,
        False,
        b_dtype,
        cutlass_torch.TensorInitType.RANDOM,
        cutlass_torch.RandomInitConfig(min_val=-10, max_val=10),
    )
    b = make_tensor_and_ref(
        b_torch_cpu,
        b_dtype,
        is_dynamic_layout=is_dynamic_layout,
        assumed_align=mixed_input_utils.get_divisibility(k),
    )

    eff_sfb_granularity_k = (
        sfb_granularity_k if sfb_granularity_k is not None else scale_granularity_k
    )
    b_scale = create_simple_sfb_scale_tensor(
        total_n,
        k,
        eff_sfb_granularity_k,
        sfb_dtype,
        is_dynamic_layout,
    )

    c_ref = cutlass_torch.matrix(1, total_n, m, False, cutlass.Float32)
    c_ref = torch.zeros_like(c_ref)
    c = make_tensor_and_ref(
        c_ref,
        c_dtype,
        is_dynamic_layout=is_dynamic_layout,
        assumed_align=16,
    )
    assert c.cute_torch is not None
    c.cute_torch.zero_()

    alignment_n = 16 * 8 // b_dtype.width
    cumsum = create_cumsum_tensor_and_ref(
        g, total_n, alignment_n, uniform_distribution=uniform_group_sizes
    )

    return GroupedMixedInputTensors(a, a_scale, b, cumsum, c, b_scale)


def run_ref_and_compare_contiguous_grouped_mixed_input_gemm_fp8(
    tensors: GroupedMixedInputTensors,
    c_dtype: type[cutlass.Numeric],
    tolerance: float,
) -> None:
    """Reference check for fp8 grouped GEMM."""
    a = tensors.a
    sfa = tensors.a_scale
    b = tensors.b
    sfb = tensors.b_scale
    cumsum = tensors.cumsum
    c = tensors.c
    assert sfa is not None, "MoE reference check requires A/SFA scales"
    assert sfb is not None, "MoE reference check requires B/SFB scales"
    assert c.cute_torch is not None

    kernel_result = c.cute_torch.cpu()
    if kernel_result.dim() == 3 and kernel_result.shape[0] == 1:
        kernel_result = kernel_result.squeeze(0)
    elif kernel_result.dim() == 3 and kernel_result.shape[2] == 1:
        kernel_result = kernel_result.squeeze(-1)

    activations_ref = b.ref_torch.to(dtype=torch.float32)
    if activations_ref.dim() == 3:
        activations_ref = (
            activations_ref.squeeze(-1)
            if activations_ref.shape[-1] == 1
            else activations_ref[:, :, 0]
        )

    sfb_ref_fp32 = sfb.ref_torch.to(dtype=torch.float32)
    assert sfb_ref_fp32.dim() == 3 and sfb_ref_fp32.shape[-1] == 1, (
        f"SFB expects shape (N_tokens, num_scales, 1), got {sfb_ref_fp32.shape}"
    )
    n_size_sfb, num_scales, _ = sfb_ref_fp32.shape
    k_size = activations_ref.shape[1]
    assert k_size % num_scales == 0, (
        f"K={k_size} not divisible by num_scales={num_scales}"
    )
    sfb_granularity_k = k_size // num_scales
    sfb_per_k = (
        sfb_ref_fp32.squeeze(-1)
        .unsqueeze(-1)
        .expand(n_size_sfb, num_scales, sfb_granularity_k)
        .reshape(n_size_sfb, k_size)
    )
    activations_ref = activations_ref * sfb_per_k

    weights_shape = a.ref_torch.shape
    weight_scales_shape = sfa.ref_torch.shape
    num_k_scales = weight_scales_shape[1]
    weights_ref = a.ref_torch.to(dtype=torch.float32).reshape(
        weights_shape[0], num_k_scales, -1, weights_shape[2]
    )
    weight_scales_ref = sfa.ref_torch.to(dtype=torch.float32).reshape(
        weight_scales_shape[0], weight_scales_shape[1], 1, weight_scales_shape[2]
    )
    weights_for_gemm = (weights_ref * weight_scales_ref).reshape(weights_shape)

    cumsum_cpu = cumsum.ref_torch
    num_tokens, m_out = activations_ref.shape[0], weights_shape[0]
    ref = torch.zeros((num_tokens, m_out), dtype=torch.float32)
    prev = 0
    for group_idx in range(1, cumsum_cpu.shape[0]):
        end = int(cumsum_cpu[group_idx])
        if end == prev:
            continue
        x_slice = activations_ref[prev:end, :]
        w_slice = weights_for_gemm[:, :, group_idx - 1]
        ref_slice = torch.einsum("mk,nk->mn", x_slice, w_slice)
        ref[prev:end, :] = ref_slice
        prev = end

    torch.testing.assert_close(kernel_result.float(), ref, atol=tolerance, rtol=1e-02)
