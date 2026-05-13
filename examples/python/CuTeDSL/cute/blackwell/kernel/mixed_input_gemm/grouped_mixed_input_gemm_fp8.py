# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""fp8 grouped mixed-input GEMM for Blackwell SM100.

This file handles both decode SFB policies. ``--sfb_dtype BFloat16`` keeps the
activation scale ``sfb`` in a plain producer layout and applies it in the
epilog. ``--sfb_dtype Float8E8M0FNU`` uses the mxfp8 hardware SFB channel via
block-scaled MMA.

The example implements a contiguous grouped/MoE GEMM. The weights are grouped
by expert while the activations are one contiguous token buffer partitioned by
``cumsum``:

  * A / ``a``: int4 weights, ``(M_out, K, G)``, transformed to fp8 in-kernel.
  * SFA / ``sfa``: bf16 weight scales, ``(M_out, K / sfa_g, G)``.
  * B / ``b``: fp8 activations, ``(N_tokens, K, 1)``, consumed directly by MMA.
  * SFB / ``sfb``: bf16 or E8M0 activation scales,
    ``(N_tokens, K / sfb_g, 1)``.
  * C / ``c``: bf16 output in user layout ``(N_tokens, M_out, 1)``.

CuTe naming follows CUTLASS GEMM axes, not ML framework naming: weights are
the A/MMA-LHS operand and activations are the B/MMA-RHS operand. The kernel
temporarily views C as ``(M_out, N_tokens, 1)`` so MMA-M/MMA-N stores land in
the user-visible token-major output.

High-level dataflow:
  1. TMA loads int4 weights A and fp8 activations B into SMEM.
  2. TRANSFORM warps convert int4 A to fp8-formatted TMEM fragments.
  3. The MMA warp runs tcgen05.mma. E8M0 SFB uses the hardware scale channel;
     BF16 SFB is applied in the epilog.
  4. Epilog warps apply the remaining software scales and biased-fp8
     correction, then store C.

To run this example from ``examples/python/CuTeDSL``:

.. code-block:: bash

    python cute/blackwell/kernel/mixed_input_gemm/grouped_mixed_input_gemm_fp8.py \
      --m 3072 --g 256 --k 2048 --n 512                                         \
      --sfa_granularity_k 256 --sfb_granularity_k 256                           \
      --a_dtype Int4 --b_dtype Float8E4M3FN --sfb_dtype BFloat16                \
      --c_dtype BFloat16 --acc_dtype Float32

With ``--mma_tiler_mnk`` omitted, the runner chooses ``128,8,K_tile`` where
``K_tile`` is the largest supported value up to 256 that satisfies the scale
alignment rules for the selected SFB dtype. For this command that is
``128,8,256``.

To collect performance with NCU:

.. code-block:: bash

    ncu --target-processes all                                                   \
      python cute/blackwell/kernel/mixed_input_gemm/grouped_mixed_input_gemm_fp8.py \
      --m 3072 --g 256 --k 2048 --n 512                                         \
      --sfa_granularity_k 256 --sfb_granularity_k 256                           \
      --warmup_iterations 1 --iterations 10 --skip_ref_check
"""

from math import log2, ceil
from typing import Optional
import argparse
import os
import sys

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.mixed_input_helpers as mixed_input_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.utils.mixed_input_helpers import TransformMode
from cutlass.cute.nvgpu import cpasync, tcgen05, OperandMajorMode

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(current_dir, "../../.."))


from blackwell.kernel.mixed_input_gemm.grouped_mixed_input_gemm_fp8_host_utils import (
    create_tensors_for_contiguous_grouped_mixed_input_gemm_fp8 as create_tensors,
    run_ref_and_compare_contiguous_grouped_mixed_input_gemm_fp8 as run_ref_and_compare,
)
from blackwell.kernel.mixed_input_gemm import (
    grouped_mixed_input_gemm_fp8_utils as fp8_utils,
)


class GroupedMixedInputGemmFp8:
    """Contiguous grouped fp8 mixed-input GEMM kernel for Blackwell SM100 (1-CTA).

    Operands (CUTLASS naming - A is the quantized LHS, regardless of ML role):
      * ``a``     : int4 weights, shape (M_out, K, G). MMA LHS; converted to
        fp8 inline by the TRANSFORM warps using the biased ``int4 + 8`` ->
        fp8 subnormal trick. Final-output correction is applied in the epilog.
      * ``sfa``   : bf16 per-(M_out x k_group x expert) weight scales. Real
        per-M-out scales applied in the epilog (the TMEM SFA channel is
        filled with unit 1.0 since bf16 doesn't fit through E8M0/E2M1).
      * ``b``     : fp8 activations, shape (N_tokens, K, 1). MMA RHS; fed
        directly into the TCU.
      * ``sfb``   : bf16 or E8M0 per-(token x k_group) activation scales.
        bf16 SFB is applied post-MMA in the epilog; E8M0 SFB is remapped into
        BSBC SMEM and consumed by HW SFB in tcgen05.mma.scale.
      * ``cumsum``: (G+1,) int32 expert-boundary offsets along the token axis,
        used by the persistent scheduler for ragged-dot dispatch.
      * ``c``     : bf16 output, shape (N_tokens, M_out, 1).

    Sibling kernels:
      * ``grouped_mixed_input_gemm_fp8_prefill.py`` - dtype-selected prefill
        path with the B rowsum moved into a precompute kernel. Use
        ``--sfb_dtype Float8E8M0FNU`` for E8M0 HW SFB.

    This kernel handles BF16 and E8M0 SFB with independent sfa/sfb
    granularities. HW SFB is unavailable for bf16, so software SFA/SFB are
    applied in the epilog. E8M0 SFB selects the blockscaled MMA policy at
    compile time. When ``--mma_tiler_mnk`` is omitted, the CLI chooses the
    largest supported MMA-K tile that satisfies the scale-alignment rules for
    the selected SFB dtype. At fine scales, the epilog runs more
    per-MMA-tile rescale events; chunked rescale defers the SFA multiply and
    SHFL reduction to the SFA-chunk boundary.

    Architectural features:
      * **Single TRANSFORM warp group** - one 4-warp TRANSFORM group processes
        BOTH M-tiles per k_tile serially. 12 warps total: 4 epilog + TMA +
        MMA + scale_TMA + scheduler + 4 transform.
      * **Fused A-TMA** - both M-tiles' A-loads share one
        ``a_load2trans_pipeline`` mbarrier with ``tx_count = 2 x a_copy_size``.
        TRANSFORM waits on ONE barrier per k_tile covering both tiles.
      * **Dual M-tile per CTA** - each CTA work-tile processes
        ``2 x mma_tiler[0]`` rows of M_out, sharing the B-load and scale-load
        across the two tiles.
      * **Unified acc / scale_load2accu pipelines** - both tiles' acc TMEM
        regions and SFA scale buffers share one pipeline barrier each.
      * **LOP3-fused biased int4->fp8 conversion** - the ``(src ^ 0x88) & 0x0F``
        and ``(src >> 4) ^ 0x08) & 0x0F`` patterns are emitted as single
        ``lop3.b32`` instructions (LUT 0x28).
      * **per-MMA-tile MMA via accumulate=False + running f32 acc** - each
        MMA tile uses ``accumulate=False`` and produces a fresh per-tile s32
        acc. BF16 SFB is folded in post-MMA; E8M0 SFB is supplied through the
        HW blockscaled MMA scale channel.
      * **chunked-rescale epilog** (key perf optimization at small mma_K) -
        instead of doing the full ``sfa*(512*acc - 8*rowsum)`` correction
        per MMA tile, the epilog accumulates per-MMA-tile SFB-weighted
        partial sums in RMEM:
          chunk_acc_sum[i]            += sfb_tile * acc_per_tile[i]
          lane_rowsum_partial[n_col]  += sfb_tile * lane_local_b_sum
        On SFA-chunk boundary (every ``num_k_tiles_per_sfa`` MMA tiles), it
        SHFL-reduces lane_rowsum_partial -> s_b_rowsums and applies one
        ``running += sfa * (512*chunk_acc - 8*weighted_rowsum)`` step,
        then resets the chunk accumulators. This amortizes the SFA multiply
        and warp-tree reduction when several MMA K tiles share one SFA tile.
        When ``sfa_g == mma_K`` this collapses to the fused per-tile path.

    Parameters:
      sfa_granularity_k : K elements per weight-scale; must be a multiple
                            of ``mma_tiler_mnk[2]``.
      sfb_granularity_k : K elements per activation-scale. For BF16 SFB it
                            must be a multiple of ``mma_tiler_mnk[2]``; for
                            E8M0 SFB it may be a multiple or divisor of
                            ``mma_tiler_mnk[2]`` and must be a multiple of
                            32. Defaults to ``sfa_granularity_k``.
      acc_dtype           : accumulator dtype (typically ``cutlass.Float32``).
      mma_tiler_mnk       : (MMA-M, MMA-N, MMA-K) tile shape. CLI callers may
                            omit this and get an auto-selected K tile. Imported
                            callers should choose the largest supported
                            ``mma_K <= 256`` that satisfies the SFA/SFB
                            scale-alignment rules for the selected SFB dtype.
      cluster_shape_mn    : (M, N) cluster shape. Must be (1, 1) - this kernel
                            is strictly 1-CTA.
      group_count         : number of experts (G).
      k_per_group         : full K dimension; saved as ``self.k_total`` for
                            constexpr use.

    Known limitations:
      * Fine-grained bf16 scales are correct, including ``mma_tiler_K=32``,
        but scale 32/64 are still epilog-overhead dominated and slower than
        the coarser 128/256/512 cases.
    """

    def __init__(
        self,
        sfa_granularity_k: int,
        acc_dtype: type[cutlass.Numeric],
        mma_tiler_mnk: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int] = (1, 1),
        group_count: int = 1,
        k_per_group: int = 0,
        sfb_granularity_k: int = None,
        num_acc_stage_override: Optional[int] = None,
    ):
        """Initialize the kernel - see class docstring for parameter docs.

        Per-tile-scale variant: SFA and SFB granularities can be set
        independently. ``sfa_granularity_k`` controls the SFA period
        (K-elements per weight scale); ``sfb_granularity_k`` controls the
        SFB period (K-elements per activation scale). Each MMA tile uses
        ``accumulate=False`` and the running f32 accumulator picks up
        ``sfa*sfb*(512*acc - 8*rowsum)`` per MMA tile, allowing the two
        granularities to differ.
        """
        # Scale granularity defines how many elements share the same scale factor
        # along the M and K modes.
        self.scale_granularity_m = 1  # fixed
        self.sfa_granularity_k = sfa_granularity_k
        # Optional override of the auto-tuned accumulator-stage count. Used to
        # fit larger MMA tiles by trading accumulator-pipeline overlap for
        # TMEM budget headroom.
        self._num_acc_stage_override = num_acc_stage_override
        # SFB granularity defaults to SFA's when not specified.
        if sfb_granularity_k is None:
            sfb_granularity_k = sfa_granularity_k
        self.sfb_granularity_k = sfb_granularity_k
        # Total K (despite the parameter name "k_per_group", it's the full K
        # - the runner passes args.k here). Saved so scale_k_tile_cnt can be
        # used as a constexpr at compile time (needed for cute.make_fragment
        # / cute.range_constexpr on the SFB pre-load path).
        self.k_total = k_per_group

        # mxf8 scale factor vector size
        self.sf_vec_size = 32
        self.bsbc_min_k = self.sf_vec_size * 4
        self.sfb_tmem_margin_cols = 16

        if cutlass.const_expr(self.sfa_granularity_k % mma_tiler_mnk[2] != 0):
            raise ValueError(
                "sfa_granularity_k must be exactly multiple of CTA tile shape K"
            )

        self.group_count = group_count
        self.acc_dtype = acc_dtype
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = mma_tiler_mnk
        self.epilog_warp_id = tuple(range(4))  # warps 0-3: epilog + STORE
        self.mma_warp_id = 4  # warp 4: MMA issuer
        self.tma_warp_id = 5  # warp 5: A/B TMA
        self.scale_tma_warp_id = 6  # warp 6: SFA TMA
        self.schedule_warp_id = 7  # warp 7: persistent scheduler
        self.transform_warp_id = tuple(range(8, 12))  # warps 8-11: int4->fp8 transform

        # Register budgets. Kept at the same values as the dual kernel.
        # Per-CTA reg total drops naturally with 4 fewer warps (~64K -> ~46K),
        # but increasing num_regs_transform_warps above 144 triggers illegal
        # instruction - the compile-time reg allocation is bounded by the
        # kernel's launch config, not the freed per-CTA budget. Leave at 144.
        self.num_regs_epilogue_warps = 144
        self.num_regs_mma_warp = 80
        self.num_regs_tma_warps = 72
        self.num_regs_transform_warps = 144
        self.num_regs_schedule_warp = 64
        self.threads_per_cta = 32 * (
            max(
                (
                    self.mma_warp_id,
                    self.tma_warp_id,
                    self.scale_tma_warp_id,
                    *self.epilog_warp_id,
                    *self.transform_warp_id,
                )
            )
            + 1
        )

        # Set barrier id for cta sync, epilogue sync, tmem ptr sync, and transform sync
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            1, 32 * len(self.epilog_warp_id)
        )
        self.tmem_ptr_sync_barrier = pipeline.NamedBarrier(2, self.threads_per_cta)
        self.sched_sync_barrier = pipeline.NamedBarrier(4, 32)
        # MX policy only: MMA waits on epilog warps' cooperative load that
        # publishes sSFB_flat at work_tile start. 1 MMA + 4 epilog = 5 warps.
        self.sfb_smem_ready_barrier = pipeline.NamedBarrier(
            5, 32 * (1 + len(self.epilog_warp_id))
        )

        self.smem_buffer_align_bytes = 1024

    def _setup_attributes(self):
        """
        Set up configurations that are dependent on GEMM inputs.

        A is always assumed K-major, so the transformed-A destination is
        always TMEM (SMEM path is dead).
        """
        if cutlass.const_expr(self.is_mxscale_sfb):
            # Blockscaled MMA. SFA stays unit in TMEM and is applied post-MMA;
            # E8M0 SFB is remapped into BSBC SMEM and consumed by HW SFB.
            self.tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                cutlass.Float8E8M0FNU,
                self.sf_vec_size,
                tcgen05.CtaGroup.ONE,
                self.mma_tiler[:2],
                tcgen05.OperandSource.TMEM,
            )
        else:
            # BF16 SFB path: non-blockscaled MMA avoids the BSBC K>=128 floor.
            # Both SFA and SFB are applied in the epilogue.
            self.tiled_mma = sm100_utils.make_trivial_tiled_mma(
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.acc_dtype,
                tcgen05.CtaGroup.ONE,
                self.mma_tiler[:2],
                tcgen05.OperandSource.TMEM,
            )

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // 1,
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_tile_shape_mnk = (
            self.cluster_shape_mn[0] * self.cta_tile_shape_mnk[0],
            self.cluster_shape_mn[1] * self.cta_tile_shape_mnk[1],
            self.cta_tile_shape_mnk[2],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (self.tiled_mma.thr_id.shape,),
        )

        # This decode kernel is 1-CTA (cluster_shape_mn=(1,1)), so there is
        # no TMA multicast: num_mcast_ctas_{a,b,sfb} are always 1 and the
        # is_*_mcast flags are always False.  All mcast_mask plumbing below
        # is stripped accordingly.

        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            False,
            self.c_layout,
            self.c_dtype,
        )

        # Compute tensor memory(TMEM) columns and stages for each pipeline
        (
            self.num_load2trans_stage,
            self.num_scale_load2accu_stage,
            self.num_trans2mma_stage,
            self.num_acc_stage,
            self.num_c_stage,
            self.num_tile_info_stage,
            self.num_acc_tmem_cols,
            self.num_a_tmem_cols,
            self.num_sfa_tmem_cols,
            self.num_sfb_tmem_cols,
        ) = self._compute_stages_and_tmem_cols(
            self.tiled_mma,
            self.mma_tiler,
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.b_dtype,
            self.sfa_dtype,
            self.sfb_dtype,
            self.c_dtype,
            self.c_layout,
            self.scale_granularity_m,
            self.sfa_granularity_k,
            self.sf_vec_size,
            sfb_granularity_k=self.sfb_granularity_k,
            k_per_group=self.k_total,
            is_mxscale_sfb=self.is_mxscale_sfb,
            num_acc_stage_override=self._num_acc_stage_override,
        )

        # Align TMEM columns for allocation
        # TMEM allocation requires power-of-2 column alignment
        # and must meet minimum allocation requirements
        self.num_tmem_alloc_cols = cute.round_up(
            self.num_acc_tmem_cols
            + self.num_sfa_tmem_cols
            + self.num_sfb_tmem_cols
            + self.num_a_tmem_cols
            + (self.sfb_tmem_margin_cols if self.is_mxscale_sfb else 0),
            cute.arch.get_min_tmem_alloc_cols("sm_100"),
        )
        self.num_tmem_alloc_cols = 2 ** (ceil(log2(self.num_tmem_alloc_cols)))

        # Get smem layout for C tensor
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )

        # Get smem layouts for A and transformed A
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            self.tiled_mma, self.mma_tiler, self.a_dtype, self.num_load2trans_stage
        )

        self.transformed_a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            self.tiled_mma,
            self.mma_tiler,
            self.a_transformed_dtype,
            self.num_trans2mma_stage,
        )

        # ((M_SHARING_SCALE, NUM_SCALES_M),(K_SHARING_SCALE, NUM_SCALES_K), STAGES)
        (
            self.scale_tile_shape,
            self.smem_layout_scale_per_stage,
            self.smem_layout_scale_staged,
        ) = mixed_input_utils.get_smem_layout_scale(
            self.mma_tiler,
            False,
            self.scale_granularity_m,
            self.sfa_granularity_k,
            self.scale_major_mode,
            self.sfa_dtype,
            self.num_scale_load2accu_stage,
        )

        # Get smem layouts for B
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            self.tiled_mma, self.mma_tiler, self.b_dtype, self.num_load2trans_stage
        )

        # Get smem layout for SFA (used to help with tmem layout).
        # The BSBC atom has K=sf_vec_size*4=128 elements. When mma_tiler_K<128
        # (e.g. mma_K=64 for sfb=64, mma_K=32 for sfb=32) the sub-atom
        # tile_to_shape produced a degenerate layout that generated illegal
        # TMEM addresses in the SFA unit-1.0 fill prologue
        # (`cudaErrorIllegalAddress`). Pad mma_K up to 128 for the BSBC
        # layout - costs a few extra TMEM cols but avoids the crash. SFA is
        # filled with unit 1.0 once at prologue and never read as data, so
        # over-allocation is correctness-neutral.
        _bsbc_min_k = self.sf_vec_size * 4
        _sfa_layout_mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            cute.round_up(self.mma_tiler[2], _bsbc_min_k),
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            self.tiled_mma,
            _sfa_layout_mma_tiler,
            self.sf_vec_size,
            1,  # Single stage only
        )
        self.sfa_smem_layout_per_stage = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        if cutlass.const_expr(self.is_mxscale_sfb):
            self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
                self.tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                1,  # Single stage
            )
            self.sfb_smem_layout_per_stage = cute.slice_(
                self.sfb_smem_layout_staged, (None, None, None, 0)
            )
        else:
            # BF16 SFB is staged in a plain epilogue SMEM tensor. These aliases
            # keep the kernel signature stable; the BSBC SFB layout is unused.
            self.sfb_smem_layout_staged = self.sfa_smem_layout_staged
            self.sfb_smem_layout_per_stage = self.sfa_smem_layout_per_stage

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        sfa: cute.Tensor,
        b: cute.Tensor,
        sfb: cute.Tensor,
        cumsum: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """
        Compile and launch the contiguous grouped fp8 mixed-input GEMM.

        Calling convention (CUTLASS-native - no boundary swap):
          ``a``      : int4 weights (MMA LHS, transformed in-kernel).
                       Shape (M_out, K, G).
          ``sfa``    : bf16 per-(M_out x k_group x expert) weight scales.
          ``b``      : fp8 activations (MMA RHS, direct to MMA).
                       Shape (N_tokens, K, 1).
          ``sfb``    : bf16 or E8M0 per-(token x k_group) activation scales.
                       bf16 is applied post-MMA; E8M0 selects HW SFB.
          ``cumsum`` : (G+1,) int32 expert-boundary offsets along the token
                       axis. ``cumsum[g+1] - cumsum[g]`` gives expert ``g``'s
                       token count.
          ``c``      : bf16 output, allocated as (N_tokens, M_out, 1) in user
                       memory; the kernel internally views this transposed
                       so MMA writes (MMA-M = M_out, MMA-N = N_tokens) land
                       in the user's expected layout.

        Note on naming: ML frameworks often call activations "A" and weights
        "B"; this kernel follows the CUTLASS convention where ``A`` is the
        quantized operand (int4 weights). The runner and the
        fp8 grouped tensor helper return tensors in this kernel-native
        order - no internal swap is performed.
        """
        # Transpose the C tensor view so kernel writes to (MMA-M = M_out,
        # MMA-N = N_tokens) land in user's (N_tokens, M_out) memory layout.
        _c_shape = c.shape
        _c_stride = c.stride
        c = cute.make_tensor(
            c.iterator,
            cute.make_layout(
                (_c_shape[1], _c_shape[0], _c_shape[2]),
                stride=(_c_stride[1], _c_stride[0], _c_stride[2]),
            ),
        )

        self.a_dtype: type[cutlass.Numeric] = a.element_type
        self.b_dtype: type[cutlass.Numeric] = b.element_type
        # Treat the converted A as raw Int8 bits; the MMA still interprets
        # them as fp8 E4M3 via the tiled_mma dtype setting. Declaring the
        # post-transform buffer as Int8 avoids lowering issues for mxf8 in
        # the vector dialect.
        self.a_transformed_dtype = cutlass.Int8
        self.sfa_dtype: type[cutlass.Numeric] = sfa.element_type
        self.sfb_dtype: type[cutlass.Numeric] = sfb.element_type
        self.is_mxscale_sfb = self.sfb_dtype == cutlass.Float8E8M0FNU
        # The TMEM scale channels that the blockscaled MMA reads are E8M0.
        # BF16 SFB keeps the hardware SFB channel at unit 1.0 and applies SFB
        # post-MMA; E8M0 SFB remaps real SFB into this channel per MMA tile.
        self.sf_mma_dtype: type[cutlass.Numeric] = cutlass.Float8E8M0FNU
        self.sf_mma_vec_size = (
            32  # this is the scale vector size for the MMA (fixed for mxf8)
        )

        self.c_dtype: type[cutlass.Numeric] = c.element_type

        # Dtype asserts for the kernel's native convention.
        if cutlass.const_expr(self.a_dtype != cutlass.Int4):
            raise ValueError(f"Invalid dtype for A {self.a_dtype}")
        if cutlass.const_expr(self.sfa_dtype != cutlass.BFloat16):
            raise ValueError(f"Invalid dtype for SFA {self.sfa_dtype}")
        if cutlass.const_expr(
            self.b_dtype not in [cutlass.Float8E5M2, cutlass.Float8E4M3FN]
        ):
            raise ValueError(f"Invalid dtype for B {self.b_dtype}")
        if cutlass.const_expr(
            self.sfb_dtype not in [cutlass.BFloat16, cutlass.Float8E8M0FNU]
        ):
            raise ValueError(
                f"Unsupported SFB dtype {self.sfb_dtype}; expected "
                "BFloat16 or Float8E8M0FNU"
            )
        if cutlass.const_expr(self.is_mxscale_sfb):
            if cutlass.const_expr(self.mma_tiler[2] < self.bsbc_min_k):
                raise ValueError(
                    f"mxscale HW-SFB requires mma_tiler_K >= {self.bsbc_min_k}"
                )
            if cutlass.const_expr(
                self.mma_tiler[2] % self.sfb_granularity_k != 0
                and self.sfb_granularity_k % self.mma_tiler[2] != 0
            ):
                raise ValueError(
                    "sfb_granularity_k must be co-aligned with mma_tiler_K"
                )
            if cutlass.const_expr(self.sfb_granularity_k % self.sf_vec_size != 0):
                raise ValueError(
                    f"sfb_granularity_k must be a multiple of sf_vec_size={self.sf_vec_size}"
                )
        elif cutlass.const_expr(self.sfb_granularity_k % self.mma_tiler[2] != 0):
            raise ValueError("sfb_granularity_k must be a multiple of mma_tiler_K")

        if cutlass.const_expr(self.c_dtype not in [cutlass.Float32, cutlass.BFloat16]):
            raise ValueError(f"Invalid dtype for C {self.c_dtype}")

        self.acc_dtype = cutlass.Float32

        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        self.scale_major_mode = utils.LayoutEnum.from_tensor(sfa).mma_major_mode()
        self.gmem_layout_scale = mixed_input_utils.get_gmem_layout_scale(
            a.shape,
            self.scale_granularity_m,
            self.sfa_granularity_k,
            self.scale_major_mode,
        )
        # SFB has native (N_tokens, num_scales, L=1) layout.
        # No broadcast layout needed since the kernel epilog indexes SFB
        # directly per-(n_global, k_chunk, 0).

        # Set up attributes that depend on runtime tensor dtypes/layouts.
        self._setup_attributes()

        # SFB is bf16 with simple (N_tokens, num_scales, L=1)
        # layout. Skip the BlockScaledBasicChunk recast - keep the native
        # iterator and rebind to gmem_layout_sfb at TMA atom creation time.

        # Set up gmem copy atoms for A (1-CTA / non-multicast):
        a_op = mixed_input_utils.get_tma_atom_kind(False, False, is_b=False)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Set up gmem copy atoms for B (1-CTA / non-multicast):
        b_op = mixed_input_utils.get_tma_atom_kind(False, False, is_b=True)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        a_scale_op = a_op  # same as A

        # Partition smem layout for scale tensor to make it compatible with TMA atom
        # ((MMA_M, MMA_K), REST_M, REST_K)
        scale_smem_layout = cute.get(
            self.tiled_mma._thrfrg_A(self.smem_layout_scale_per_stage.outer), mode=[1]
        )
        # ((MMA_M, MMA_K), REST_M, REST_K)
        scale_smem_layout = cute.dice(
            scale_smem_layout,
            (1, (1,) * cute.rank(self.smem_layout_scale_per_stage.outer)),
        )
        tma_atom_scale, tma_tensor_scale = cute.nvgpu.make_tiled_tma_atom_A(
            a_scale_op,
            cute.make_tensor(sfa.iterator, self.gmem_layout_scale),
            scale_smem_layout,
            # (SCALE_M, 1, SCALE_K)
            (self.scale_tile_shape[0], 1, self.scale_tile_shape[1]),
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if sfa.element_type is cutlass.Float32 else None
            ),
        )

        # Calculate copy size for tensor A, B, scale
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        scale_copy_size = cute.size_in_bytes(self.sfa_dtype, scale_smem_layout)

        self.num_tma_load_bytes_a = a_copy_size
        # SFB is loaded by epilog warps via cp.async at work_tile prologue
        # so load2mma_pipeline carries only B.
        self.num_tma_load_bytes_b = b_copy_size * cute.size(self.tiled_mma.thr_id.shape)
        # Scale pipeline carries 2 x SFA (tile-0 + tile-1).
        self.num_tma_load_bytes_scale = scale_copy_size
        self.tile_sched_params, grid = fp8_utils.compute_persistent_grid(
            c,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
            m_tile_multiplier=2,
        )

        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c,
            epi_smem_layout,
            self.epi_tile,
        )

        # Pipeline stage counts captured as locals so the struct annotations
        # stay readable. Each pipeline needs a full/empty mbarrier pair.
        _n_load2trans = self.num_load2trans_stage
        _n_scale = self.num_scale_load2accu_stage
        _n_trans2mma = self.num_trans2mma_stage
        _n_acc = self.num_acc_stage
        _n_tile_info = self.num_tile_info_stage
        _Int64 = cutlass.Int64
        _MemRange = cute.struct.MemRange

        @cute.struct
        class SharedStorage:
            # Scheduler work-tile info (producer: scheduler warp; consumers: all).
            tile_info: _MemRange[cutlass.Int32, 4 * _n_tile_info]
            tile_info_full_mbar_ptr: _MemRange[_Int64, _n_tile_info]
            tile_info_empty_mbar_ptr: _MemRange[_Int64, _n_tile_info]

            # load2trans - A/B TMA -> SMEM (both tiles share one mbar pair).
            a_load2trans_full_mbar_ptr: _MemRange[_Int64, _n_load2trans]
            a_load2trans_empty_mbar_ptr: _MemRange[_Int64, _n_load2trans]

            # scale_load2accu - SFA TMA -> epilog.  Both tiles share one
            # mbarrier with tx_count = 2 x scale_bytes, so the pair of
            # per-k_tile scale TMAs complete the same barrier.
            a_scale_load2accu_full_mbar_ptr: _MemRange[_Int64, _n_scale]
            a_scale_load2accu_empty_mbar_ptr: _MemRange[_Int64, _n_scale]

            # trans2mma - TRANSFORM warps -> MMA (unified across tiles).
            # Transform writes BOTH tiles' TMEM, then arrives ONCE; MMA waits
            # ONCE, issues tcgen05.mma for each tile, releases ONCE.
            a_trans2mma_full_mbar_ptr: _MemRange[_Int64, _n_trans2mma]
            a_trans2mma_empty_mbar_ptr: _MemRange[_Int64, _n_trans2mma]

            # b_load2mma - B TMA -> MMA (shares num_load2trans_stage depth with A).
            b_load2mma_full_mbar_ptr: _MemRange[_Int64, _n_load2trans]
            b_load2mma_empty_mbar_ptr: _MemRange[_Int64, _n_load2trans]

            # acc - MMA -> epilog (unified across tiles).
            # MMA commits ONCE per k-block (both tiles done); epilog waits ONCE
            # and consumes both tiles' acc TMEM regions before releasing.
            acc_full_mbar_ptr: _MemRange[_Int64, _n_acc]
            acc_empty_mbar_ptr: _MemRange[_Int64, _n_acc]

            # TMEM allocator handshake.
            tmem_dealloc_mbar: _Int64
            tmem_holding_buf: cutlass.Int32

        self.shared_storage = SharedStorage

        self.kernel(
            self.tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_scale,
            tma_tensor_scale,
            tma_atom_b,
            tma_tensor_b,
            sfb,
            tma_atom_c,
            tma_tensor_c,
            c,
            cumsum,
            self.group_count,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.transformed_a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.smem_layout_scale_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tensor_c: cute.Tensor,
        cumsum: cute.Tensor,
        group_count: cutlass.Constexpr[int],
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout: cute.ComposedLayout,
        a_smem_layout_transform: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        scale_smem_layout: cute.ComposedLayout,
        sfb_smem_layout: cute.Layout,
        c_smem_layout: cute.ComposedLayout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        """
        Device entry point for the persistent grouped GEMM.

        Important tensor views on entry:
          mA_mkl   : A weights in GMEM, shape (M_out, K, G)
          mSFA_mkl : A/SFA scales in GMEM, shape (M_out, K / sfa_g, G)
          mB_nkl   : B activations in GMEM, shape (N_tokens, K, 1)
          mSFB_nkl : B/SFB scales in GMEM, shape (N_tokens, K / sfb_g, 1)
          mC_mnl   : C output view used by MMA, shape (M_out, N_tokens, 1)

        CuTe suffix convention used below:
          g* tensors are CTA-local GMEM tiles produced by cute.local_tile.
          tCg* tensors are per-MMA-thread partitions of those GMEM tiles.
          s* tensors live in SMEM; tCr*/tCt* tensors are register/TMEM views.
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        # num_k_tiles_per_scale: how many K-dimension CTA tiles share one scale value.
        # E.g. sfa_granularity_k=256, cta_tile_K=128 -> 2 K-tiles per scale.
        num_k_tiles_per_sfa = self.sfa_granularity_k // self.cta_tile_shape_mnk[2]

        # Prefetch TMA descriptors
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)
            cpasync.prefetch_descriptor(tma_atom_sfa)

        mma_tile_coord_v = cutlass.Int32(0)
        is_leader_cta = True
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Initialize load2transform pipeline, which tracks the dependencies between TMA's loading
        # of A and B, and the transformation of A and MMA's consumption
        transform_thread_idx = (
            tidx - 32 * self.transform_warp_id[0]
            if tidx >= 32 * self.transform_warp_id[0]
            else tidx
        )
        # Fused-TMA variant: ONE pipeline covers BOTH tile TMAs. tx_count is
        # 2x a_copy_size because two cp.async.bulk.tensor calls (tile-0 and
        # tile-1) arrive on this single mbarrier each stage.
        a_load2trans_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.a_load2trans_full_mbar_ptr.data_ptr(),
            num_stages=self.num_load2trans_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                4,  # one TRANSFORM warp group = 4 warps per stage
            ),
            tx_count=2 * self.num_tma_load_bytes_a,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=transform_thread_idx,
            mcast_mode_mn=(1, 0),
            defer_sync=True,
        )

        # Initialize the unified scale_load2accu pipeline. Tracks SFA TMA
        # load -> epilog scale consume. Both tile-0 and tile-1 SFA TMAs share
        # this pipeline: ONE producer_acquire covers both, the shared
        # mbarrier's tx_count is 2 x SFA_bytes so both TMAs must land to
        # complete it.
        # SFB is NOT a TMA - loaded via cp.async by the
        # same warp; its arrival is gated by cp_async_wait_all / fence.
        _scale_tx = 2 * self.num_tma_load_bytes_scale
        scale_load2accu_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.a_scale_load2accu_full_mbar_ptr.data_ptr(),
            num_stages=self.num_scale_load2accu_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(self.epilog_warp_id),
            ),
            tx_count=_scale_tx,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            mcast_mode_mn=(1, 0),  # multicast for sfa will only happen on the M-mode
            defer_sync=True,
        )

        # Initialize pipeline for tensor B and SFB load to MMA.
        # PipelineTmaMultiConsumersAsync keeps the B SMEM stage valid until BOTH
        # the MMA consumer (TCGen05Mma) and the epilog consumer (AsyncThread) have
        # released it.  The epilog reads B SMEM to compute in-kernel column sums.
        load2mma_pipeline = pipeline.PipelineTmaMultiConsumersAsync.create(
            barrier_storage=storage.b_load2mma_full_mbar_ptr.data_ptr(),
            num_stages=self.num_load2trans_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group_umma=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group_async=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.epilog_warp_id)
            ),
            tx_count=self.num_tma_load_bytes_b,
            cta_layout_vmnk=cluster_layout_vmnk,
            mcast_mode_mn=(0, 1),  # multicast for B will only happen on the N-mode
            defer_sync=True,
        )

        # Initialize unified trans2mma pipeline. The TRANSFORM warps convert BOTH
        # M-tiles per k_tile, emit ONE fence_view_async_tmem_store, then commit
        # ONCE. The MMA warp waits ONCE and issues two tcgen05.mma's (one per
        # tile) before releasing. Saves 2 barrier ops per k_tile vs the
        # per-tile pipelines, at the cost of losing the tile-0-early-MMA /
        # tile-1-transform overlap.
        trans2mma_pipeline = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.a_trans2mma_full_mbar_ptr.data_ptr(),
            num_stages=self.num_trans2mma_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * 4,  # TRANSFORM warps = 128 threads per stage
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Initialize unified accumulator pipeline. MMA commits ONCE per k-block
        # after both tile-0 and tile-1 tcgen05.mma's have been issued; the epilog
        # warps wait ONCE, consume BOTH tiles' acc TMEM regions, and release
        # ONCE. Saves 2 barrier ops per k-block vs per-tile pipelines.
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.epilog_warp_id)
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Initialize tile info pipeline, which tracks the dependencies between
        # tile scheduling warp and other warps
        # Skip scheduler warp when computing consumer thread count
        num_tile_info_pipeline_consumer_threads = self.threads_per_cta - 32
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tile_info_full_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_info_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                num_tile_info_pipeline_consumer_threads,
            ),
            defer_sync=True,
        )

        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_ptr_sync_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # --- Allocate SMEM tensors ---
        # SMEM staging tensors. Shapes are described in tile coordinates:
        #   sA_tile{0,1}: raw int4 A for each of the two M tiles.
        #   sSFA_tile{0,1}: bf16 SFA for each M tile.
        #   sB: fp8 B tile shared by both M tiles.
        #   sSFB_post: bf16 SFB tile in plain token-major layout.
        #   sC: epilog staging before the TMA/SIMT store to C.
        sC = smem.allocate_tensor(
            element_type=self.c_dtype,
            layout=c_smem_layout.outer,
            byte_alignment=self.smem_buffer_align_bytes,
            swizzle=c_smem_layout.inner,
        )
        sA_tile0 = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout.outer,
            byte_alignment=self.smem_buffer_align_bytes,
            swizzle=a_smem_layout.inner,
        )
        # tile-1 A SMEM tensor - same layout as sA_tile0, holds the second
        # M-tile's raw int4 weights for the dual-tile pipeline.
        sA_tile1 = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout.outer,
            byte_alignment=self.smem_buffer_align_bytes,
            swizzle=a_smem_layout.inner,
        )
        sSFA_tile0 = smem.allocate_tensor(
            element_type=self.sfa_dtype,
            layout=scale_smem_layout.outer,
            byte_alignment=self.smem_buffer_align_bytes,
            swizzle=scale_smem_layout.inner,
        )
        # tile-1 SMEM scale tensor - same layout as sSFA_tile0, holds the
        # second M-tile's scale factors. Loaded by the scale TMA warp in the
        # unified `scale_load2accu_pipeline` (same mbarrier as tile-0).
        sSFA_tile1 = smem.allocate_tensor(
            element_type=self.sfa_dtype,
            layout=scale_smem_layout.outer,
            byte_alignment=self.smem_buffer_align_bytes,
            swizzle=scale_smem_layout.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout.outer,
            byte_alignment=self.smem_buffer_align_bytes,
            swizzle=b_smem_layout.inner,
        )
        if cutlass.const_expr(self.is_mxscale_sfb):
            sSFB_bsbc = smem.allocate_tensor(
                element_type=self.sfb_dtype,
                layout=sfb_smem_layout,
                byte_alignment=self.smem_buffer_align_bytes,
            )
            _sfb_flat_n = self.cta_tile_shape_mnk[1]
            _sfb_flat_chunks = self.k_total // self.sfb_granularity_k
            sSFB_flat = smem.allocate_tensor(
                element_type=cutlass.Float32,
                layout=cute.make_layout(
                    (_sfb_flat_n, _sfb_flat_chunks),
                    stride=(1, _sfb_flat_n),
                ),
                byte_alignment=128,
            )
            sSFB_post = sSFB_flat
        else:
            # BF16 SFB staging in plain token-major SMEM. Two guard rows let
            # full odd-start tiles stage one aligned bf16 pair before the true
            # tile and read with a +1 row offset.
            _bf16_sfb_n = self.cta_tile_shape_mnk[1]
            _bf16_sfb_stage_n = _bf16_sfb_n + 2
            _bf16_sfb_chunks = self.k_total // self.sfb_granularity_k
            sSFB_post = smem.allocate_tensor(
                element_type=self.sfb_dtype,
                layout=cute.make_layout(
                    (_bf16_sfb_stage_n, _bf16_sfb_chunks),
                    stride=(1, _bf16_sfb_stage_n),
                ),
                byte_alignment=128,
            )
        # Scratch buffer for per-token row-sums of ACTIVATIONS (the B operand).
        #
        # In this kernel the activations are the B operand (MMA RHS, fp8,
        # direct to MMA) - NOT the weights. `s_b_rowsums[i]` = sum_K(B[i, :])
        # for token i in this CTA tile.
        #
        # Layout: one float32 per token in the CTA tile, sized by cta_tile_n
        # (N-axis = tokens). 4 epilog warps split the tokens evenly; lane 0
        # of each warp stores its token's sum here.
        #
        # Used by the biased-fp8 correction:
        #     corrected[t, m] = scale[t, m] * (512*acc[t, m] - 8*b_rowsum[t])
        # where t in [0, cta_tile_n).
        s_b_rowsums = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((self.cta_tile_shape_mnk[1],)),
            byte_alignment=128,
        )

        sTile_info = storage.tile_info.get_tensor(
            cute.make_layout((4, self.num_tile_info_stage), stride=(1, 4))
        )

        # 1-CTA (cluster_shape_mn=(1,1)) -> no TMA multicast; all TMA copies
        # below use the plain non-multicast form (mcast_mask omitted).

        # ---- Global tiles: convert full GMEM tensors into CTA tile views ----
        # The last three modes are loopM/loopK/loopL or loopN/loopK/loopL;
        # the persistent scheduler later selects concrete loop coordinates.

        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bM, bK, loopM, loopK, loopL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, loopN, loopK, loopL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # SFB is read from sSFB_post (loaded by cp.async at work_tile start);
        # the epilog using indexing - no local_tile needed. mSFB_nkl has
        # native shape (N_tokens, num_scales, L=1).
        # (bM, bN, loopM, loopN, loopL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        gC_mnl_simt = cute.local_tile(
            tensor_c, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])  # true k tile count

        # ---- MMA-thread partitions of CTA-local GMEM tiles ----
        # partition_A/B/C maps the CTA tile into the per-thread view expected
        # by the tiled MMA atom and by the associated TMA/transform helpers.
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, loopM, loopK, loopL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_M, MMA_K, loopM, loopK, loopL)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (MMA, MMA_N, MMA_K, loopN, loopK, loopL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # No tCgSFB - SFB is read from sSFB_post in the epilog k_tile loop
        # mSFB_nkl indexing in the epilog (per-thread LDG).
        # (MMA, MMA_M, MMA_N, loopM, loopN, loopL)
        tCgC = thr_mma.partition_C(gC_mnl)
        tCgC_simt = thr_mma.partition_C(gC_mnl_simt)

        # Setup copy atom to load A from shared memory for further transformation
        copy_atom_a_input = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.a_dtype, num_bits_per_copy=32
        )
        a_smem_shape = tiled_mma.partition_shape_A(
            cute.dice(self.mma_tiler, (1, None, 1))
        )
        # Setup copy atom to store transformed A into tensor memory or shared memory
        copy_atom_a_transform = mixed_input_utils.get_copy_atom_a_transform(
            self.tiled_mma.op.a_dtype,
            False,
            tcgen05.OperandSource.TMEM,
            a_smem_shape,
            self.a_dtype,
        )

        copy_atom_sfa = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.sfa_dtype, num_bits_per_copy=32
        )
        copy_atom_sfa_expanded = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.sfa_dtype,
            num_bits_per_copy=32,
        )

        # Partition global/shared tensor for TMA load A/B
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )

        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tAsA_tile0, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA_tile0, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # tile-1 TMA partition. Same atom/coord/layout as tile 0;
        # only the SMEM destination (sA_tile1) differs. Produces tAsA_tile1 that
        # is indexed identically to tAsA_tile0 - the per-tile distinction is the
        # underlying SMEM buffer.
        tAsA_tile1, _ = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA_tile1, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )

        thr_mma_leader_cta = tiled_mma.get_slice(0)
        # (MMA, MMA_M, MMA_K, STAGE)
        tCsSFA_tile0 = thr_mma_leader_cta.partition_A(sSFA_tile0)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tAsSFA_tile0, tAgSFA = mixed_input_utils.scale_tma_partition(
            tCsSFA_tile0,
            tCgSFA,
            tma_atom_sfa,
            block_in_cluster_coord_vmnk,
            a_cta_layout,
        )
        # tile-1 scale TMA partition. Same atom/coord/layout as
        # tile 0; only the SMEM destination (sSFA_tile1) differs. tAgSFA is
        # unchanged - the per-tile GMEM M-offset is applied at indexing time
        # via `2*cta_coord_m + ni_tile`.
        tCsSFA_tile1 = thr_mma_leader_cta.partition_A(sSFA_tile1)
        tAsSFA_tile1, _ = mixed_input_utils.scale_tma_partition(
            tCsSFA_tile1,
            tCgSFA,
            tma_atom_sfa,
            block_in_cluster_coord_vmnk,
            a_cta_layout,
        )

        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # No SFB TMA partition - SFB is loaded via cp.async by the epilog warps
        # in scale_tma_warp using simple thread-strided GMEM->SMEM copies.

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #

        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        #
        # Alloc tensor memory buffer
        #
        tmem.allocate(self.num_tmem_alloc_cols)
        tmem.wait_for_alloc()

        #
        # Retrieving tensor memory ptr and make accumulator tensor
        #
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

        #
        # Retrieving tensor memory ptr and make accumulator/SFA/SFB tensor
        #

        # Make accumulator tmem tensor
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_base_tile0 = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        # tile-1 acc TMEM view. Starts at offset num_acc_tmem_cols // 2
        # (second half of the doubled acc TMEM region).
        tCtAcc_base_tile1 = cute.make_tensor(
            tmem_ptr + self.num_acc_tmem_cols // 2,
            tCtAcc_fake.layout,
        )

        # Make transformed A tensor in TMEM (tile-0 and tile-1 halves).
        # (MMA, MMA_M, MMA_K, STAGE)
        tmem_ptr_transform_tile0 = cute.recast_ptr(
            tCtAcc_base_tile0.iterator + self.num_acc_tmem_cols,
            dtype=self.tiled_mma.op.a_dtype,
        )
        tCrA_tile0 = cute.make_tensor(
            tmem_ptr_transform_tile0,
            tiled_mma.make_fragment_A(a_smem_layout_transform.outer).layout,
        )

        # tile-1 A TMEM view. Starts at offset num_acc_tmem_cols +
        # num_a_tmem_cols // 2 (after tile-0 A region). group-1 writes transformed
        # tile-1 weights here; MMA reads it for the tile-1 cute.gemm.
        tmem_ptr_transform_tile1 = cute.recast_ptr(
            tCtAcc_base_tile0.iterator
            + self.num_acc_tmem_cols
            + self.num_a_tmem_cols // 2,
            dtype=self.tiled_mma.op.a_dtype,
        )
        tCrA_tile1 = cute.make_tensor(
            tmem_ptr_transform_tile1,
            tiled_mma.make_fragment_A(a_smem_layout_transform.outer).layout,
        )

        # SFA TMEM sits after the transformed-A region.
        sfa_tmem_offset = self.num_acc_tmem_cols + self.num_a_tmem_cols

        # Make SFA tmem tensor
        # (MMA, MMA_M, MMA_K)
        sfa_tmem_ptr = cute.recast_ptr(
            tmem_ptr + sfa_tmem_offset,
            dtype=self.sf_mma_dtype,
        )
        tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_mma_vec_size,
            self.sfa_smem_layout_per_stage,
        )
        tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

        # Make SFB tmem tensor. BF16 SFB keeps this channel at unit 1.0 and
        # applies SFB post-MMA; MX remaps E8M0 SFB into this channel per tile.
        sfb_tmem_offset = (
            sfa_tmem_offset
            + self.num_sfa_tmem_cols
            + (self.sfb_tmem_margin_cols if self.is_mxscale_sfb else 0)
        )
        sfb_tmem_ptr = cute.recast_ptr(
            tmem_ptr + sfb_tmem_offset,
            dtype=self.sf_mma_dtype,
        )
        _sfb_tmem_smem_layout = (
            self.sfb_smem_layout_per_stage
            if self.is_mxscale_sfb
            else self.sfa_smem_layout_per_stage
        )
        tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_mma_vec_size,
            _sfb_tmem_smem_layout,
        )
        tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

        if cutlass.const_expr(self.is_mxscale_sfb):
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t,
                tCtSFB_compact_s2t,
            ) = fp8_utils.mainloop_s2t_copy_and_partition(
                sSFB_bsbc,
                tCtSFB,
                self.sf_mma_dtype,
                tcgen05.CtaGroup.ONE,
            )

        #
        # Fill unit scale 1.0 for SFA mma loop. Per-tile-scale variant:
        # SFB TMEM is also unit 1.0 (real SFB is applied post-MMA in the
        # epilog).
        #

        if self.transform_warp_id[0] <= warp_idx <= self.transform_warp_id[-1]:
            # Remove stride of 0 elements for store
            tCtSFA_filtered = cute.filter_zeros(tCtSFA)
            rmem_sfa = cute.make_fragment_like(tCtSFA_filtered)

            # Setup RMEM to TMEM copy over the threads of this warps
            r2t_copy_atom = cute.make_copy_atom(
                tcgen05.St32x32bOp(tcgen05.Repetition(1)), tCtSFA_filtered.element_type
            )
            r2t_copy = tcgen05.make_tmem_copy(r2t_copy_atom, tCtSFA_filtered)

            thr_copy_t2r = r2t_copy.get_slice(tidx)
            thrA = thr_copy_t2r.partition_S(rmem_sfa)
            thtA = thr_copy_t2r.partition_D(tCtSFA_filtered)

            # Initalize the threads registers to unit scale and copy.
            thrA.fill(1.0)
            cute.copy(r2t_copy, thrA, thtA)

            if cutlass.const_expr(not self.is_mxscale_sfb):
                # BF16 policy: fill SFB TMEM with unit 1.0 once at prologue.
                tCtSFB_filtered = cute.filter_zeros(tCtSFB)
                rmem_sfb = cute.make_fragment_like(tCtSFB_filtered)
                r2t_copy_atom_b = cute.make_copy_atom(
                    tcgen05.St32x32bOp(tcgen05.Repetition(1)),
                    tCtSFB_filtered.element_type,
                )
                r2t_copy_b = tcgen05.make_tmem_copy(r2t_copy_atom_b, tCtSFB_filtered)
                thr_copy_t2r_b = r2t_copy_b.get_slice(tidx)
                thrB = thr_copy_t2r_b.partition_S(rmem_sfb)
                thtB = thr_copy_t2r_b.partition_D(tCtSFB_filtered)
                thrB.fill(1.0)
                cute.copy(r2t_copy_b, thrB, thtB)
        cute.arch.fence_view_async_tmem_store()

        if cutlass.const_expr(self.is_mxscale_sfb):
            # Safety-pad unused BSBC cells with E8M0 1.0.
            if self.transform_warp_id[0] <= warp_idx <= self.transform_warp_id[-1]:
                sSFB_bsbc_filt = cute.filter_zeros(sSFB_bsbc)
                sSFB_bsbc_i32 = cute.recast_tensor(sSFB_bsbc_filt, cutlass.Int32)
                _bsbc_size_i32: cutlass.Constexpr[int] = cute.size(sSFB_bsbc_i32)
                _num_xform_thr: cutlass.Constexpr[int] = 32 * len(
                    self.transform_warp_id
                )
                _per_thr_fills: cutlass.Constexpr[int] = (
                    _bsbc_size_i32 + _num_xform_thr - 1
                ) // _num_xform_thr
                transform_local_tidx = tidx - 32 * self.transform_warp_id[0]
                for _idx in cutlass.range_constexpr(_per_thr_fills):
                    _gid = transform_local_tidx + _idx * _num_xform_thr
                    if _gid < _bsbc_size_i32:
                        sSFB_bsbc_i32[_gid] = 0x7F7F7F7F

        #
        # Schedule warp
        #
        if warp_idx == self.schedule_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_schedule_warp)
            fp8_utils.produce_grouped_tile_info(
                tile_sched_params,
                bidx,
                bidy,
                bidz,
                block_in_cluster_coord_vmnk,
                tile_info_pipeline,
                sTile_info,
                self.sched_sync_barrier,
                self.num_tile_info_stage,
                self.cluster_tile_shape_mnk,
                self.cluster_shape_mn,
                self.cta_tile_shape_mnk,
                group_count,
                cumsum,
            )

        #
        # Specialized TMA load warp for A/B/SFB tensors
        #
        if warp_idx == self.tma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_tma_warps)
            # Persistent tile scheduling loop
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_info_stage
            )
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            work_tile = mixed_input_utils.make_contiguous_group_work_tile_info(
                group_count, sTile_info[(None, tile_info_consumer_state.index)]
            )
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()
            a_load2trans_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_load2trans_stage
            )
            load2mma_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_load2trans_stage
            )
            while work_tile.is_valid_tile:
                # grid halved - cta_coord_m is at 2 x cta_tile_m granularity.
                # Tile 0 is even kernel-M tile, tile 1 is odd.
                _cm2 = (work_tile.cta_coord_m // 1) * 2
                tAgA_slice_tile0 = tAgA[(None, _cm2, None, work_tile.group_idx)]
                tAgA_slice_tile1 = tAgA[(None, _cm2 + 1, None, work_tile.group_idx)]
                # Apply offset to B tensor based on group search result
                coord_n_offset = (
                    (work_tile.coord_n, 0, 0)
                    if cutlass.const_expr(self.b_major_mode == OperandMajorMode.MN)
                    else (0, work_tile.coord_n, 0)
                )
                tBgB_slice = cute.make_tensor(
                    (
                        tBgB.iterator[0] + coord_n_offset[0],
                        tBgB.iterator[1] + coord_n_offset[1],
                        tBgB.iterator[2] + coord_n_offset[2],
                    ),
                    # ((atom_v, rest_v), RestK)
                    cute.slice_(tBgB.layout, (None, 0, None, 0)),
                )

                # No SFB TMA load - SFB is cp.async-loaded by the epilog warps
                # (alongside SFA loads). Warp 5 only loads A and B here.

                a_load2trans_producer_state.reset_count()
                a_peek_load2trans_empty_status = cutlass.Boolean(1)
                if a_load2trans_producer_state.count < k_tile_cnt:
                    a_peek_load2trans_empty_status = (
                        a_load2trans_pipeline.producer_try_acquire(
                            a_load2trans_producer_state
                        )
                    )
                load2mma_producer_state.reset_count()
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Fused-TMA: one producer_acquire + two TMA copies on same
                    # mbar. Barrier's expect_tx=2*a_copy_size so it completes
                    # after BOTH tile A TMAs land.
                    a_load2trans_pipeline.producer_acquire(
                        a_load2trans_producer_state, a_peek_load2trans_empty_status
                    )
                    load2mma_pipeline.producer_acquire(load2mma_producer_state)
                    # Tile-0 A TMA.
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice_tile0[(None, a_load2trans_producer_state.count)],
                        tAsA_tile0[(None, a_load2trans_producer_state.index)],
                        tma_bar_ptr=a_load2trans_pipeline.producer_get_barrier(
                            a_load2trans_producer_state
                        ),
                    )
                    # Tile-1 A TMA - same mbar as tile-0.
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice_tile1[(None, a_load2trans_producer_state.count)],
                        tAsA_tile1[(None, a_load2trans_producer_state.index)],
                        tma_bar_ptr=a_load2trans_pipeline.producer_get_barrier(
                            a_load2trans_producer_state
                        ),
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, load2mma_producer_state.count)],
                        tBsB[(None, load2mma_producer_state.index)],
                        tma_bar_ptr=load2mma_pipeline.producer_get_barrier(
                            load2mma_producer_state
                        ),
                    )
                    # No SFB TMA - load2mma_pipeline carries only B
                    # (alongside SFA). load2mma_pipeline carries only B now.
                    load2mma_pipeline.producer_commit(load2mma_producer_state)
                    a_load2trans_producer_state.advance()
                    load2mma_producer_state.advance()
                    if a_load2trans_producer_state.count < k_tile_cnt:
                        a_peek_load2trans_empty_status = (
                            a_load2trans_pipeline.producer_try_acquire(
                                a_load2trans_producer_state
                            )
                        )

                # Advance to next tile
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                work_tile = mixed_input_utils.make_contiguous_group_work_tile_info(
                    group_count, sTile_info[(None, tile_info_consumer_state.index)]
                )
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            # Wait A/B/SFB buffer empty
            a_load2trans_pipeline.producer_tail(a_load2trans_producer_state)
            load2mma_pipeline.producer_tail(load2mma_producer_state)

        # Specialized TMA load for SFA tensors
        if warp_idx == self.scale_tma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_tma_warps)

            # Persistent tile scheduling loop
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_info_stage
            )
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            work_tile = mixed_input_utils.make_contiguous_group_work_tile_info(
                group_count, sTile_info[(None, tile_info_consumer_state.index)]
            )
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            # Unified scale producer state - one acquire covers both tiles'
            # scale TMAs, which hit the same mbarrier (tx_count=2 x bytes).
            scale_load2accu_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_scale_load2accu_stage
            )
            scale_k_tile_cnt = cute.size(mSFA_mkl.layout.shape[1][1])
            while work_tile.is_valid_tile:
                # ((atom_v, rest_v), RestK)
                # grid halved on M. cta_coord_m is at
                # 2 x cta_tile_m granularity. Tile 0 scales are at even kernel-M tile.
                tAgSFA_slice_tile0 = tAgSFA[
                    (
                        None,
                        2 * (work_tile.cta_coord_m // 1),
                        None,
                        work_tile.group_idx,
                    )
                ]

                # Filter zeros in rest mode
                rest_filtered_tile0 = cute.filter_zeros(
                    tAgSFA_slice_tile0[(0, None)].layout
                )
                tAgSFA_slice_filtered = cute.make_tensor(
                    tAgSFA_slice_tile0.iterator,
                    cute.make_layout(
                        (tAgSFA_slice_tile0.layout[0].shape, rest_filtered_tile0.shape),
                        stride=(
                            tAgSFA_slice_tile0.layout[0].stride,
                            rest_filtered_tile0.stride,
                        ),
                    ),
                )

                # tile-1 GMEM slice for the odd kernel-M tile
                # (2*cta_coord_m + 1). Same K-filter pattern as tile 0.
                tAgSFA_slice_tile1 = tAgSFA[
                    (
                        None,
                        2 * (work_tile.cta_coord_m // 1) + 1,
                        None,
                        work_tile.group_idx,
                    )
                ]
                rest_filtered_tile1 = cute.filter_zeros(
                    tAgSFA_slice_tile1[(0, None)].layout
                )
                tAgSFA_slice_tile1_filtered = cute.make_tensor(
                    tAgSFA_slice_tile1.iterator,
                    cute.make_layout(
                        (
                            tAgSFA_slice_tile1.layout[0].shape,
                            rest_filtered_tile1.shape,
                        ),
                        stride=(
                            tAgSFA_slice_tile1.layout[0].stride,
                            rest_filtered_tile1.stride,
                        ),
                    ),
                )

                # No SFB load here - SFB is loaded into sSFB_post by the epilog warps
                # from GMEM in the epilog (epilog warp/thread per-thread LDG).

                scale_load2accu_producer_state.reset_count()
                peek_scale_load2accu_empty_status = cutlass.Boolean(1)
                if scale_load2accu_producer_state.count < scale_k_tile_cnt:
                    peek_scale_load2accu_empty_status = (
                        scale_load2accu_pipeline.producer_try_acquire(
                            scale_load2accu_producer_state
                        )
                    )
                for k_tile in cutlass.range(0, scale_k_tile_cnt, 1, unroll=1):
                    # ONE acquire covers BOTH tiles' scale TMAs - the
                    # shared mbarrier's tx_count = 2 x scale_bytes, so both
                    # TMAs must arrive to complete it.
                    scale_load2accu_pipeline.producer_acquire(
                        scale_load2accu_producer_state,
                        peek_scale_load2accu_empty_status,
                    )
                    # Tile-0 scale TMA.
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice_filtered[
                            (None, scale_load2accu_producer_state.count)
                        ],
                        tAsSFA_tile0[(None, scale_load2accu_producer_state.index)],
                        tma_bar_ptr=scale_load2accu_pipeline.producer_get_barrier(
                            scale_load2accu_producer_state
                        ),
                    )
                    # Tile-1 scale TMA - same mbarrier.
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice_tile1_filtered[
                            (None, scale_load2accu_producer_state.count)
                        ],
                        tAsSFA_tile1[(None, scale_load2accu_producer_state.index)],
                        tma_bar_ptr=scale_load2accu_pipeline.producer_get_barrier(
                            scale_load2accu_producer_state
                        ),
                    )
                    scale_load2accu_producer_state.advance()
                    peek_scale_load2accu_empty_status = cutlass.Boolean(1)
                    if scale_load2accu_producer_state.count < scale_k_tile_cnt:
                        peek_scale_load2accu_empty_status = (
                            scale_load2accu_pipeline.producer_try_acquire(
                                scale_load2accu_producer_state
                            )
                        )
                # Advance to next tile
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                work_tile = mixed_input_utils.make_contiguous_group_work_tile_info(
                    group_count, sTile_info[(None, tile_info_consumer_state.index)]
                )
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            # Wait scale buffers to be empty (unified - one tail covers both tiles).
            scale_load2accu_pipeline.producer_tail(scale_load2accu_producer_state)

        # Specialized TRANSFORM warps - fused-TMA + unified-trans2mma dual-tile.
        # group-0 (warps 8-11) processes tile-0 AND tile-1 per k_tile serially.
        # Per k_tile the body does:
        #   - ONE consumer_wait on a_load2trans_pipeline (both tiles' SMEM landed)
        #   - ONE producer_acquire on trans2mma_pipeline (covers both tiles' TMEM A)
        #   - Transform tile-0 -> write TMEM A tile-0 (in-place store)
        #   - Transform tile-1 -> write TMEM A tile-1
        #   - ONE fence_view_async_tmem_store covering both stores
        #   - ONE trans2mma producer_commit signals BOTH tiles ready
        #   - ONE a_load2trans consumer_release frees the shared SMEM stage
        if self.transform_warp_id[0] <= warp_idx <= self.transform_warp_id[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_transform_warps)
            transform_local_tidx = (
                tidx - 32 * self.transform_warp_id[0]
            )  # [0, 128) per group

            # Per-tile partitions - destination is always TMEM.
            src_copy_a0, dst_copy_a0, tAsA_input0, tAsA_transform0 = (
                mixed_input_utils.transform_partition(
                    tcgen05.OperandSource.TMEM,
                    TransformMode.ConvertScale,
                    copy_atom_a_input,
                    copy_atom_a_transform,
                    sA_tile0,
                    tCrA_tile0,
                    transform_local_tidx,
                )
            )
            src_copy_a1, dst_copy_a1, tAsA_input1, tAsA_transform1 = (
                mixed_input_utils.transform_partition(
                    tcgen05.OperandSource.TMEM,
                    TransformMode.ConvertScale,
                    copy_atom_a_input,
                    copy_atom_a_transform,
                    sA_tile1,
                    tCrA_tile1,
                    transform_local_tidx,
                )
            )

            tArA_load0 = cute.make_rmem_tensor(
                cute.append(tAsA_input0[(None, None, None, None, 0)].shape, 1),
                tAsA_input0.element_type,
            )
            tArA_transform0 = cute.make_rmem_tensor(
                cute.append(tAsA_input0[(None, None, None, None, 0)].shape, 1),
                self.a_transformed_dtype,
            )
            tArA_load1 = cute.make_rmem_tensor(
                cute.append(tAsA_input1[(None, None, None, None, 0)].shape, 1),
                tAsA_input1.element_type,
            )
            tArA_transform1 = cute.make_rmem_tensor(
                cute.append(tAsA_input1[(None, None, None, None, 0)].shape, 1),
                self.a_transformed_dtype,
            )
            transform_tiler_size = min(
                cute.size(cute.coalesce(tAsA_input0.layout), mode=[0]), 32
            )
            transform_tiler = cute.make_layout(transform_tiler_size)

            # Persistent scheduling: pull the first work-tile descriptor.
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_info_stage
            )
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            work_tile = mixed_input_utils.make_contiguous_group_work_tile_info(
                group_count, sTile_info[(None, tile_info_consumer_state.index)]
            )
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            # Single consumer state shared across tiles (both tiles' SMEM
            # arrival in one mbar stage).
            a_load2trans_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_load2trans_stage
            )
            # Unified trans2mma producer state - single barrier signals both
            # tiles' TMEM A delivery.
            trans2mma_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_trans2mma_stage
            )

            while work_tile.is_valid_tile:
                a_load2trans_consumer_state.reset_count()
                trans2mma_producer_state.reset_count()

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # ONE wait covers both tiles' SMEM delivery for this stage.
                    a_load2trans_pipeline.consumer_wait(a_load2trans_consumer_state)

                    # Acquire the shared trans2mma slot BEFORE touching either
                    # tile's TMEM A region (both tiles write into the same stage).
                    trans2mma_pipeline.producer_acquire(trans2mma_producer_state)

                    # ---------- Tile-0 transform ----------
                    tAsA_input_slice0 = fp8_utils.divide_tensor_by_tiler(
                        tAsA_input0[
                            (None, None, None, None, a_load2trans_consumer_state.index)
                        ],
                        transform_tiler,
                    )
                    tArA_load_slice0 = fp8_utils.divide_tensor_by_tiler(
                        tArA_load0[(None, None, None, None, 0)],
                        transform_tiler,
                    )
                    tArA_transform_buffer0 = tArA_transform0[
                        (None, None, None, None, 0)
                    ]
                    tArA_transform_slice0 = fp8_utils.divide_tensor_by_tiler(
                        tArA_transform_buffer0, transform_tiler
                    )

                    for idx in cutlass.range_constexpr(
                        cute.size(tArA_load_slice0, mode=[1])
                    ):
                        cute.autovec_copy(
                            tAsA_input_slice0[(None, idx)],
                            tArA_load_slice0[(None, idx)],
                        )

                    for idx in cutlass.range_constexpr(
                        cute.size(tArA_load_slice0, mode=[1])
                    ):
                        tensor_transformed = mixed_input_utils.cvt_tensor_a_biased(
                            tArA_load_slice0[(None, idx)],
                            self.a_transformed_dtype,
                        )
                        tArA_transform_slice0[(None, idx)].store(tensor_transformed)

                    mixed_input_utils.store_transformed_a(
                        tArA_transform_buffer0,
                        tAsA_transform0[
                            (None, None, None, None, trans2mma_producer_state.index)
                        ],
                        dst_copy_a0,
                    )

                    # ---------- Tile-1 transform ----------
                    tAsA_input_slice1 = fp8_utils.divide_tensor_by_tiler(
                        tAsA_input1[
                            (None, None, None, None, a_load2trans_consumer_state.index)
                        ],
                        transform_tiler,
                    )
                    tArA_load_slice1 = fp8_utils.divide_tensor_by_tiler(
                        tArA_load1[(None, None, None, None, 0)],
                        transform_tiler,
                    )
                    tArA_transform_buffer1 = tArA_transform1[
                        (None, None, None, None, 0)
                    ]
                    tArA_transform_slice1 = fp8_utils.divide_tensor_by_tiler(
                        tArA_transform_buffer1, transform_tiler
                    )

                    for idx in cutlass.range_constexpr(
                        cute.size(tArA_load_slice1, mode=[1])
                    ):
                        cute.autovec_copy(
                            tAsA_input_slice1[(None, idx)],
                            tArA_load_slice1[(None, idx)],
                        )

                    for idx in cutlass.range_constexpr(
                        cute.size(tArA_load_slice1, mode=[1])
                    ):
                        tensor_transformed = mixed_input_utils.cvt_tensor_a_biased(
                            tArA_load_slice1[(None, idx)],
                            self.a_transformed_dtype,
                        )
                        tArA_transform_slice1[(None, idx)].store(tensor_transformed)

                    mixed_input_utils.store_transformed_a(
                        tArA_transform_buffer1,
                        tAsA_transform1[
                            (None, None, None, None, trans2mma_producer_state.index)
                        ],
                        dst_copy_a1,
                    )

                    # ONE fence covers BOTH tiles' TMEM stores before the
                    # unified commit; ONE commit signals MMA that both tiles'
                    # TMEM A regions are ready.
                    cute.arch.fence_view_async_tmem_store()
                    trans2mma_pipeline.producer_commit(trans2mma_producer_state)
                    trans2mma_producer_state.advance()

                    # ONE release covers both tiles' SMEM; the TMA warp can
                    # now refill this stage for the next k_tile.
                    a_load2trans_pipeline.consumer_release(a_load2trans_consumer_state)
                    a_load2trans_consumer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                work_tile = mixed_input_utils.make_contiguous_group_work_tile_info(
                    group_count, sTile_info[(None, tile_info_consumer_state.index)]
                )
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            # No explicit producer_tail: MMA consumes ALL produced stages
            # within each work-tile before the tile_info barrier releases,
            # so no drain is needed.

        # Specialized MMA warp
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma_warp)
            # Persistent tile scheduling loop
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_info_stage
            )
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            work_tile = mixed_input_utils.make_contiguous_group_work_tile_info(
                group_count, sTile_info[(None, tile_info_consumer_state.index)]
            )
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()
            # Unified trans2mma consumer state - single barrier signals BOTH
            # tiles' TMEM A are ready (transform warps commit once per k_tile).
            trans2mma_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_trans2mma_stage
            )
            load2mma_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_load2trans_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            while work_tile.is_valid_tile:
                load2mma_consumer_state.reset_count()
                trans2mma_consumer_state.reset_count()
                if cutlass.const_expr(self.is_mxscale_sfb):
                    self.sfb_smem_ready_barrier.arrive_and_wait()
                peek_trans2mma_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    if trans2mma_consumer_state.count < k_tile_cnt:
                        peek_trans2mma_full_status = (
                            trans2mma_pipeline.consumer_try_wait(
                                trans2mma_consumer_state
                            )
                        )
                    # Per-tile-scale: each MMA tile uses accumulate=False and
                    # commits the acc pipeline ONCE per MMA tile. The epilog
                    # rescales by sfa*sfb*(512*acc - 8*rowsum) per tile and
                    # accumulates into a register-resident running f32. There
                    # is no longer a "k_block" outer loop.
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        # ONE acquire covers BOTH tiles' acc TMEM regions
                        # for THIS k_tile; commit signals both tiles done at
                        # the end of this MMA tile.
                        acc_pipeline.producer_acquire(acc_producer_state)
                        # (MMA, MMA_M, MMA_N)
                        tCtAcc_tile0 = tCtAcc_base_tile0[
                            (None, None, None, acc_producer_state.index)
                        ]
                        tCtAcc_tile1 = tCtAcc_base_tile1[
                            (None, None, None, acc_producer_state.index)
                        ]

                        # wait for a tile of data to arrive from producers.
                        # ONE trans2mma wait covers BOTH tiles' TMEM A delivery.
                        load2mma_pipeline.consumer_wait(load2mma_consumer_state)
                        trans2mma_pipeline.consumer_wait(trans2mma_consumer_state)

                        kblock_coord_a = (
                            None,
                            None,
                            None,
                            trans2mma_consumer_state.index,
                        )
                        kblock_coord_b = (
                            None,
                            None,
                            None,
                            load2mma_consumer_state.index,
                        )

                        if cutlass.const_expr(self.is_mxscale_sfb):
                            self._remap_sfb_flat_to_bsbc(
                                sSFB_flat, sSFB_bsbc, tidx, k_tile
                            )
                            s2t_stage_coord = (
                                None,
                                None,
                                None,
                                None,
                                0,
                            )
                            tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[
                                s2t_stage_coord
                            ]
                            cute.copy(
                                tiled_copy_s2t_sfb,
                                tCsSFB_compact_s2t_staged,
                                tCtSFB_compact_s2t,
                            )
                            cute.gemm(
                                tiled_mma,
                                tCtAcc_tile0,
                                [tCrA_tile0[kblock_coord_a], tCtSFA],
                                [tCrB[kblock_coord_b], tCtSFB],
                                tCtAcc_tile0,
                            )
                            cute.gemm(
                                tiled_mma,
                                tCtAcc_tile1,
                                [tCrA_tile1[kblock_coord_a], tCtSFA],
                                [tCrB[kblock_coord_b], tCtSFB],
                                tCtAcc_tile1,
                            )
                        else:
                            cute.gemm(
                                tiled_mma,
                                tCtAcc_tile0,
                                tCrA_tile0[kblock_coord_a],
                                tCrB[kblock_coord_b],
                                tCtAcc_tile0,
                            )
                            cute.gemm(
                                tiled_mma,
                                tCtAcc_tile1,
                                tCrA_tile1[kblock_coord_a],
                                tCrB[kblock_coord_b],
                                tCtAcc_tile1,
                            )

                        # ONE trans2mma release covers both tiles' TMEM A.
                        trans2mma_pipeline.consumer_release(trans2mma_consumer_state)
                        trans2mma_consumer_state.advance()

                        # release load2mma AFTER both tile gemms have
                        # been issued. The TCGen05Mma release fires when ALL prior
                        # tcgen05 ops on this SM have retired - covering both tile-0
                        # and tile-1 gemms. If we released after tile-0 only, TMA
                        # could refill B SMEM while tile-1 MMA was still reading it.
                        load2mma_pipeline.consumer_release(
                            load2mma_consumer_state,
                            pipeline.PipelineOp.TCGen05Mma,
                        )
                        load2mma_consumer_state.advance()

                        peek_trans2mma_full_status = cutlass.Boolean(1)
                        if trans2mma_consumer_state.count < k_tile_cnt:
                            peek_trans2mma_full_status = (
                                trans2mma_pipeline.consumer_try_wait(
                                    trans2mma_consumer_state
                                )
                            )

                        # Per-tile commit - both tiles' fresh acc ready.
                        acc_pipeline.producer_commit(acc_producer_state)
                        acc_producer_state.advance()

                # Advance to next tile
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                work_tile = mixed_input_utils.make_contiguous_group_work_tile_info(
                    group_count, sTile_info[(None, tile_info_consumer_state.index)]
                )
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            # Wait for accumulator buffer empty
            acc_pipeline.producer_tail(acc_producer_state)

        # Specialized acc update and epilogue warps
        if warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_epilogue_warps)
            epi_tidx = tidx
            # Construct scale tensor view as C
            scale_view_as_C_layout = cute.make_layout(
                (
                    scale_smem_layout.outer[0].shape,
                    self.cta_tile_shape_mnk[1],
                    scale_smem_layout.outer[2].shape,
                ),
                stride=(
                    scale_smem_layout.outer[0].stride,
                    0,
                    scale_smem_layout.outer[2].stride,
                ),
            )
            scale_view_as_C_tile0 = cute.make_tensor(
                sSFA_tile0.iterator,
                scale_view_as_C_layout,
            )
            # tile-1 SMEM view of scale tensor used for partitioning.
            # Same layout as scale_view_as_C_tile0; only the underlying SMEM iterator
            # (sSFA_tile1) differs.
            scale_view_as_C_tile1 = cute.make_tensor(
                sSFA_tile1.iterator,
                scale_view_as_C_layout,
            )
            # Partition for epilogue and accumulator update
            (
                tiled_copy_t2r_tile0,
                tTR_tAcc_base_tile0,
                tTR_rAcc,
                tTR_rAcc_final_tile0,
                tTR_sScale_tile0,
            ) = fp8_utils.epilog_and_acc_update_tmem_copy_and_partition(
                epi_tidx,
                tCtAcc_base_tile0,
                tCgC,
                scale_view_as_C_tile0,
                epi_tile,
                self.cta_tile_shape_mnk,
                self.c_layout,
                self.c_dtype,
                self.acc_dtype,
                False,
            )
            # tile-1 partition. Only the TMEM source differs
            # (tCtAcc_base_tile1). tTR_rAcc_final_tile1 is an independent RMEM
            # tensor so we accumulate into two separate register banks.
            # tile-1 now also uses its own SMEM scale view
            # (scale_view_as_C_tile1) so tTR_sScale_tile1 reads from sSFA_tile1.
            (
                tiled_copy_t2r_tile1,
                tTR_tAcc_base_tile1,
                _tTR_rAcc_tile1,
                tTR_rAcc_final_tile1,
                tTR_sScale_tile1,
            ) = fp8_utils.epilog_and_acc_update_tmem_copy_and_partition(
                epi_tidx,
                tCtAcc_base_tile1,
                tCgC,
                scale_view_as_C_tile1,
                epi_tile,
                self.cta_tile_shape_mnk,
                self.c_layout,
                self.c_dtype,
                self.acc_dtype,
                False,
            )

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)

            # ---------------------------------------------------------------
            # Register-local activation-row-sum setup. Barrier-free pattern
            # where each epilog thread owns one N-col (one token) and stores
            # that token's rowsum to s_b_rowsums directly.
            #
            # Each epilog thread handles exactly one MMA-N/token column (M-major TMEM
            # partition).  We compute ONE row-sum for that token directly from
            # sB using a dynamic N-index - no SMEM write, no bar.sync, and
            # only cta_k reads (not cta_n x cta_k).
            #
            # sB shape: ((cta_n, 32), 1, (cta_k//32), stages)
            # coord:    ((n, k%32),   0,  k//32,       stage)
            # ---------------------------------------------------------------
            _cta_n = self.cta_tile_shape_mnk[1]
            _cta_k = self.cta_tile_shape_mnk[2]

            # Epilog B pipeline state: one entry per B TMA stage consumed
            _load2mma_epilog_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_load2trans_stage
            )
            # Deferred-release state: advances in sync with _load2mma_epilog_state
            # but releases are issued after correction.
            _load2mma_epilog_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_load2trans_stage
            )
            # ---------------------------------------------------------------

            tiled_copy_r2s, tRS_rC, tRS_sC = (
                mixed_input_utils.epilog_smem_copy_and_partition(
                    self.c_layout,
                    self.c_dtype,
                    self.acc_dtype,
                    tiled_copy_t2r_tile0,
                    tTR_rC,
                    epi_tidx,
                    sC,
                )
            )
            (tma_atom_c, bSG_sC, bSG_gC_partitioned, simt_atom, tTR_gC_partitioned) = (
                mixed_input_utils.epilog_gmem_copy_and_partition(
                    self.c_dtype,
                    epi_tidx,
                    tma_atom_c,
                    tiled_copy_t2r_tile0,
                    tCgC,
                    tCgC_simt,
                    epi_tile,
                    sC,
                )
            )

            # Predicates
            thr_mapping = cute.make_identity_tensor(
                (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1])
            )
            thr_mapping_mn = cute.flat_divide(thr_mapping, epi_tile)
            thr_copy_t2r = tiled_copy_t2r_tile0.get_slice(epi_tidx)
            m_thr_offset = thr_copy_t2r.partition_D(thr_mapping_mn)
            m_thr_offset = cute.group_modes(m_thr_offset, 3, cute.rank(m_thr_offset))

            # NOTE: do NOT precompute N-cols per subtile here (outside the
            # k_tile loop).  m_thr_offset element accesses outside a DSL loop
            # context return compile-time Python ints (incorrect for our
            # purposes).  Instead, _n_sub is computed inside the subtile loop
            # where the MLIR context is active, yielding a correct runtime
            # ArithValue.  See comment below near the correction loop.

            # Unified acc consumer state - single barrier signals BOTH tiles'
            # acc TMEM are ready.
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            # Unified scale consumer state - single barrier signals BOTH
            # tiles' SFA SMEM are ready.
            scale_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_scale_load2accu_stage,
            )

            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )
            # cp.async copy atom for per-thread 32-bit SFB GMEM->SMEM
            # load.  Source/dest are recast to Int32 at the call site;
            # one int32 = 2 bf16 elements packed along N.
            _sfb_cpasync_atom = cute.make_copy_atom(
                cpasync.CopyG2SOp(),
                cutlass.Int32,
                num_bits_per_copy=32,
            )
            # Persistent tile scheduling loop
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_info_stage
            )
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            work_tile = mixed_input_utils.make_contiguous_group_work_tile_info(
                group_count, sTile_info[(None, tile_info_consumer_state.index)]
            )
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()
            num_prev_subtiles = cutlass.Int32(0)
            scale_k_tile_cnt = cute.size(mSFA_mkl.layout.shape[1][1])
            while work_tile.is_valid_tile:
                # perform accumulator update with scales
                tTR_rAcc_final_tile0.fill(0.0)
                # reset tile-1 RMEM accumulator too.
                tTR_rAcc_final_tile1.fill(0.0)

                tTR_rScale_tile0 = cute.make_rmem_tensor(
                    cute.slice_(tTR_sScale_tile0, (None, None, None, 0, None, 0)).shape,
                    self.sfa_dtype,
                )
                # tile-1 RMEM scale tensor - populated from
                # tTR_sScale_tile1 which is partitioned on sSFA_tile1.
                tTR_rScale_tile1 = cute.make_rmem_tensor(
                    cute.slice_(tTR_sScale_tile1, (None, None, None, 0, None, 0)).shape,
                    self.sfa_dtype,
                )
                _sfb_num_chunks: cutlass.Constexpr[int] = (
                    self.k_total // self.sfb_granularity_k
                )
                _sfb_n: cutlass.Constexpr[int] = self.cta_tile_shape_mnk[1]
                _num_epilog_thr: cutlass.Constexpr[int] = 32 * len(self.epilog_warp_id)
                _sfb_smem_row_offset = cutlass.Int32(0)
                if cutlass.const_expr(self.is_mxscale_sfb):
                    _mx_sfb_total: cutlass.Constexpr[int] = _sfb_n * _sfb_num_chunks
                    _mx_per_thr_loads: cutlass.Constexpr[int] = (
                        _mx_sfb_total + _num_epilog_thr - 1
                    ) // _num_epilog_thr
                    for _mx_li in cutlass.range_constexpr(_mx_per_thr_loads):
                        _mx_idx = epi_tidx + _mx_li * _num_epilog_thr
                        if _mx_idx < _mx_sfb_total:
                            _mx_chunk_idx = _mx_idx // _sfb_n
                            _mx_n_local = _mx_idx % _sfb_n
                            _mx_n_global = work_tile.coord_n + _mx_n_local
                            if (_mx_n_local < work_tile.distance_to_boundary) and (
                                _mx_n_global < mSFB_nkl.shape[0]
                            ):
                                sSFB_post[(_mx_n_local, _mx_chunk_idx)] = mSFB_nkl[
                                    (_mx_n_global, _mx_chunk_idx, 0)
                                ].to(cutlass.Float32)
                            else:
                                sSFB_post[(_mx_n_local, _mx_chunk_idx)] = (
                                    cutlass.Float32(1.0)
                                )
                    self.sfb_smem_ready_barrier.arrive_and_wait()
                else:
                    _sfb_smem_row_offset = fp8_utils.stage_bf16_sfb_gmem_to_smem(
                        mSFB_nkl,
                        sSFB_post,
                        _sfb_cpasync_atom,
                        epi_tidx,
                        work_tile.coord_n,
                        work_tile.distance_to_boundary,
                        _num_epilog_thr,
                        _sfb_num_chunks,
                        _sfb_n,
                        False,
                    )
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    self.epilog_sync_barrier.arrive_and_wait()

                # -------------------------------------------------------
                # Per-MMA-tile B-row-sum + acc-rescale loop.
                #
                # For each MMA tile (= cta_tile_k K-elements):
                #   1. Compute per-MMA-tile rowsum (consumes ONE B stage).
                #   2. Wait for the per-MMA-tile s32 acc TMEM (acc pipeline
                #      now fires per MMA tile - accumulate=False MMA).
                #   3. On SFA chunk boundary, autovec_copy the SFA SMEM ->
                #      RMEM register fragment (kept across MMA tiles within
                #      a chunk). On SFB chunk boundary, the SFB index advances
                #      into sSFB_post automatically via integer division.
                #   4. running_f32 += sfa*sfb*(512*acc_per_tile - 8*rowsum_per_tile)
                #
                # Total acc-pipeline events: k_tile_cnt (NOT scale_k_tile_cnt).
                # -------------------------------------------------------
                _load2mma_epilog_state.reset_count()
                _load2mma_epilog_release_state.reset_count()
                scale_consumer_state.reset_count()
                acc_consumer_state.reset_count()
                if cutlass.const_expr(num_k_tiles_per_sfa != 1):
                    peek_scale_full_status = cutlass.Boolean(1)
                    if scale_consumer_state.count < scale_k_tile_cnt:
                        peek_scale_full_status = (
                            scale_load2accu_pipeline.consumer_try_wait(
                                scale_consumer_state
                            )
                        )
                    peek_acc_full_status = cutlass.Boolean(1)
                    if acc_consumer_state.count < k_tile_cnt:
                        peek_acc_full_status = acc_pipeline.consumer_try_wait(
                            acc_consumer_state
                        )
                # Number of MMA tiles per SFB chunk. Used to index sSFB_post
                # per MMA tile.
                num_k_tiles_per_sfb_local = (
                    self.sfb_granularity_k // self.cta_tile_shape_mnk[2]
                )
                # ---------------------------------------------------------
                # CHUNKED RESCALE - defer SHFL + SFA-mul + running update
                # to the SFA chunk boundary.
                #
                # Per-MMA-tile work (cheap):
                #   * lane_rowsum_partial[_nw] += sfb_tile * sum_lanes_local B
                #   * chunk_acc_sum[i]         += sfb_tile * acc_per_tile[i]
                # Per-SFA-chunk work (expensive, fires every num_k_tiles_per_sfa):
                #   * 5-shfl reduce -> s_b_rowsums (publishes weighted rowsum)
                #   * running[i] += sfa[i] * (512*chunk_acc - 8*weighted_rowsum)
                #   * reset chunk_acc_sum and lane_rowsum_partial
                #
                # When chunk = 1 tile, this reduces to the fused per-tile
                # correction path.
                # ---------------------------------------------------------
                _n_per_warp_const: cutlass.Constexpr[int] = (
                    self.cta_tile_shape_mnk[1] // 4
                )
                chunk_acc_sum_tile0 = cute.make_rmem_tensor(
                    tTR_rAcc_final_tile0.shape, self.acc_dtype
                )
                chunk_acc_sum_tile1 = cute.make_rmem_tensor(
                    tTR_rAcc_final_tile1.shape, self.acc_dtype
                )
                chunk_acc_sum_tile0.fill(0.0)
                chunk_acc_sum_tile1.fill(0.0)
                lane_rowsum_partial = cute.make_rmem_tensor(
                    cute.make_layout((_n_per_warp_const,)), self.acc_dtype
                )
                lane_rowsum_partial.fill(0.0)

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # SFB chunk index - k_tile // num_k_tiles_per_sfb.
                    _k_chunk_idx_sfb = k_tile // num_k_tiles_per_sfb_local

                    # Per-MMA-tile lane-local SFB-weighted rowsum (no SHFL
                    # - that's deferred to chunk close). sSFB_post was
                    # already published before the k_tile loop.
                    _load2mma_epilog_state = self._chunked_lane_rowsum(
                        warp_idx,
                        epi_tidx,
                        sB,
                        sSFB_post,
                        _sfb_smem_row_offset,
                        _k_chunk_idx_sfb,
                        k_tile,
                        lane_rowsum_partial,
                        load2mma_pipeline,
                        _load2mma_epilog_state,
                        _cta_n,
                        _cta_k,
                    )

                    # Per-MMA-tile acc consume + SFB-weighted accumulate.
                    if cutlass.const_expr(num_k_tiles_per_sfa == 1):
                        acc_pipeline.consumer_wait(acc_consumer_state)
                    else:
                        acc_pipeline.consumer_wait(
                            acc_consumer_state, peek_acc_full_status
                        )
                    _acc_stage_idx = acc_consumer_state.index

                    self._chunked_acc_accumulate(
                        tTR_tAcc_base_tile0,
                        tiled_copy_t2r_tile0,
                        tTR_rAcc,
                        chunk_acc_sum_tile0,
                        sSFB_post,
                        _sfb_smem_row_offset,
                        _k_chunk_idx_sfb,
                        m_thr_offset,
                        _acc_stage_idx,
                    )
                    self._chunked_acc_accumulate(
                        tTR_tAcc_base_tile1,
                        tiled_copy_t2r_tile1,
                        tTR_rAcc,
                        chunk_acc_sum_tile1,
                        sSFB_post,
                        _sfb_smem_row_offset,
                        _k_chunk_idx_sfb,
                        m_thr_offset,
                        _acc_stage_idx,
                    )

                    # Per-MMA-tile acc release.
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()
                    if cutlass.const_expr(num_k_tiles_per_sfa != 1):
                        peek_acc_full_status = cutlass.Boolean(1)
                        if acc_consumer_state.count < k_tile_cnt:
                            peek_acc_full_status = acc_pipeline.consumer_try_wait(
                                acc_consumer_state
                            )

                    # On SFA chunk close: SHFL reduce, sync, load SFA, apply
                    # running update, reset chunk accumulators, release SFA.
                    _next_k_tile = k_tile + 1
                    _at_chunk_close = (_next_k_tile % num_k_tiles_per_sfa) == 0
                    if _at_chunk_close:
                        # 1. SHFL reduce lane_rowsum_partial -> s_b_rowsums
                        #    (lane 0 publishes per-token-column weighted rowsum;
                        #    partials reset to 0 for next chunk).
                        self._chunked_close_publish_rowsum(
                            warp_idx,
                            epi_tidx,
                            lane_rowsum_partial,
                            s_b_rowsums,
                            _cta_n,
                        )
                        # 2. Sync to publish s_b_rowsums across all epilog
                        #    threads before any non-publishing thread reads it.
                        self.epilog_sync_barrier.arrive_and_wait()
                        # 3. Wait for SFA chunk and load to RMEM.
                        if cutlass.const_expr(num_k_tiles_per_sfa == 1):
                            scale_load2accu_pipeline.consumer_wait(scale_consumer_state)
                        else:
                            scale_load2accu_pipeline.consumer_wait(
                                scale_consumer_state, peek_scale_full_status
                            )
                        _scale_stage_idx = scale_consumer_state.index
                        tTR_sScale_tile0_slice = cute.slice_(
                            tTR_sScale_tile0,
                            (None, None, None, 0, None, _scale_stage_idx),
                        )
                        cute.autovec_copy(tTR_sScale_tile0_slice, tTR_rScale_tile0)
                        tTR_sScale_tile1_slice = cute.slice_(
                            tTR_sScale_tile1,
                            (None, None, None, 0, None, _scale_stage_idx),
                        )
                        cute.autovec_copy(tTR_sScale_tile1_slice, tTR_rScale_tile1)
                        # 4. Apply chunk update: running += sfa*(512*chunk - 8*weighted).
                        #    Resets chunk_acc_sum to 0 in the same pass.
                        self._chunked_close_apply(
                            chunk_acc_sum_tile0,
                            tTR_rAcc_final_tile0,
                            tTR_rScale_tile0,
                            s_b_rowsums,
                            m_thr_offset,
                        )
                        self._chunked_close_apply(
                            chunk_acc_sum_tile1,
                            tTR_rAcc_final_tile1,
                            tTR_rScale_tile1,
                            s_b_rowsums,
                            m_thr_offset,
                        )
                        # 5. SFA release.
                        scale_load2accu_pipeline.consumer_release(scale_consumer_state)
                        scale_consumer_state.advance()
                        if cutlass.const_expr(num_k_tiles_per_sfa != 1):
                            peek_scale_full_status = cutlass.Boolean(1)
                            if scale_consumer_state.count < scale_k_tile_cnt:
                                peek_scale_full_status = (
                                    scale_load2accu_pipeline.consumer_try_wait(
                                        scale_consumer_state
                                    )
                                )
                num_prev_subtiles = fp8_utils.store_decode_accumulator_tiles(
                    tTR_rAcc_final_tile0,
                    tTR_rAcc_final_tile1,
                    tTR_rC,
                    tTR_gC_partitioned,
                    tiled_copy_r2s,
                    tRS_rC,
                    tRS_sC,
                    tma_atom_c,
                    bSG_sC,
                    bSG_gC_partitioned,
                    simt_atom,
                    c_pipeline,
                    self.epilog_sync_barrier,
                    m_thr_offset,
                    warp_idx,
                    self.epilog_warp_id[0],
                    work_tile.cta_coord_m,
                    work_tile.coord_n,
                    work_tile.distance_to_boundary,
                    tensor_c.shape[0],
                    tensor_c.layout.stride[1],
                    self.cta_tile_shape_mnk[0],
                    self.cta_tile_shape_mnk[1],
                    self.num_c_stage,
                    self.c_dtype,
                    num_prev_subtiles,
                    self.c_layout.is_n_major_c(),
                )
                # Advance to next tile
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                work_tile = mixed_input_utils.make_contiguous_group_work_tile_info(
                    group_count, sTile_info[(None, tile_info_consumer_state.index)]
                )
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            c_pipeline.producer_tail()

    @cute.jit
    def _chunked_lane_rowsum(
        self,
        warp_idx: cutlass.Int32,
        epi_tidx: cutlass.Int32,
        sB: cute.Tensor,
        sSFB_post: cute.Tensor,
        sfb_smem_row_offset: cutlass.Int32,
        k_chunk_idx_sfb: cutlass.Int32,
        k_tile: cutlass.Int32,
        lane_rowsum_partial: cute.Tensor,
        load2mma_pipeline,
        _load2mma_epilog_state,
        cta_tile_n: cutlass.Constexpr[int],
        cta_tile_k: cutlass.Constexpr[int],
    ):
        """Per-MMA-tile lane-local SFB-weighted rowsum, accumulated in RMEM.

        For each (warp, N-col-this-warp-owns), each lane sums B over the K
        elements of THIS MMA tile, multiplies by the per-tile SFB scalar
        for that N-col, and accumulates into ``lane_rowsum_partial[_nw]``.
        The cross-lane SHFL reduction is DEFERRED to ``_chunked_close_*``,
        which fires once per SFA chunk (not per MMA tile) - this is the
        main pipe_fmalite saving.
        """
        _lane = epi_tidx % 32
        _n_per_warp: cutlass.Constexpr[int] = cta_tile_n // 4
        _num_k_groups: cutlass.Constexpr[int] = cta_tile_k // 32
        _num_sub: cutlass.Constexpr[int] = max(cta_tile_k // self.sfb_granularity_k, 1)
        _ko_per_sub: cutlass.Constexpr[int] = _num_k_groups // _num_sub

        load2mma_pipeline.consumer_wait(_load2mma_epilog_state)
        cute.arch.fence_proxy("async.shared", space="cta")
        _stage = _load2mma_epilog_state.index

        for _nw in cutlass.range_constexpr(_n_per_warp):
            _n_col = warp_idx * _n_per_warp + _nw
            _cs = cutlass.Float32(0.0)
            if cutlass.const_expr(self.is_mxscale_sfb):
                if cutlass.const_expr(_num_sub == 1):
                    _num_k_tiles_per_sfb: cutlass.Constexpr[int] = max(
                        self.sfb_granularity_k // cta_tile_k, 1
                    )
                    _k_chunk = k_tile // _num_k_tiles_per_sfb
                    _sfb = sSFB_post[(_n_col, _k_chunk)].to(cutlass.Float32)
                    for _ko in cutlass.range_constexpr(_num_k_groups):
                        _b = sB[((_n_col, _lane), 0, _ko, _stage)].to(cutlass.Float32)
                        _cs = _cs + _b
                    _cs = _sfb * _cs
                else:
                    _k_chunk_base = k_tile * _num_sub
                    for _b_sub in cutlass.range_constexpr(_num_sub):
                        _ko_acc = cutlass.Float32(0.0)
                        for _ks in cutlass.range_constexpr(_ko_per_sub):
                            _ko = _b_sub * _ko_per_sub + _ks
                            _b = sB[((_n_col, _lane), 0, _ko, _stage)].to(
                                cutlass.Float32
                            )
                            _ko_acc = _ko_acc + _b
                        _sfb = sSFB_post[(_n_col, _k_chunk_base + _b_sub)].to(
                            cutlass.Float32
                        )
                        _cs = _cs + _sfb * _ko_acc
            else:
                for _ko in cutlass.range_constexpr(_num_k_groups):
                    _b = sB[((_n_col, _lane), 0, _ko, _stage)].to(cutlass.Float32)
                    _cs = _cs + _b
                _sfb = sSFB_post[(_n_col + sfb_smem_row_offset, k_chunk_idx_sfb)].to(
                    cutlass.Float32
                )
                _cs = _sfb * _cs
            lane_rowsum_partial[(_nw,)] = lane_rowsum_partial[(_nw,)] + _cs

        load2mma_pipeline.consumer_release(
            _load2mma_epilog_state,
            pipeline.PipelineOp.AsyncThread,
        )
        _load2mma_epilog_state.advance()
        return _load2mma_epilog_state

    @cute.jit
    def _chunked_acc_accumulate(
        self,
        tTR_tAcc_base: cute.Tensor,
        tiled_copy_t2r,
        tTR_rAcc: cute.Tensor,
        chunk_acc_sum: cute.Tensor,
        sSFB_post: cute.Tensor,
        sfb_smem_row_offset: cutlass.Int32,
        k_chunk_idx_sfb: cutlass.Int32,
        m_thr_offset: cute.Tensor,
        acc_stage_idx: cutlass.Int32,
    ):
        """Per-MMA-tile T2R + SFB-weighted acc accumulation into RMEM.

        Replaces the per-MMA-tile portion of _correct_tile. Per element:
            chunk_acc_sum[i] += sfb_i * acc_per_tile[i]
        The sfa-mul, the (512*acc - 8*rowsum) combination, and the +=running
        update are all deferred to _chunked_close_apply. Saves 4 of the 5
        FMAs per element per MMA tile.
        """
        tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_idx)]
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])

        for subtile_idx in cutlass.range_constexpr(subtile_cnt):
            tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
            chunk_acc_subtile = chunk_acc_sum[(None, None, None, subtile_idx)]
            acc_vec = tTR_rAcc.load()
            m_thr_sub_k = m_thr_offset[(None, None, None, subtile_idx)]
            for i in cutlass.range(
                cute.size(chunk_acc_subtile.shape), unroll_full=True
            ):
                if cutlass.const_expr(self.is_mxscale_sfb):
                    chunk_acc_subtile[i] = chunk_acc_subtile[(i)] + acc_vec[(i)]
                else:
                    n_local_i = m_thr_sub_k[(i)][1]
                    sfb_i = sSFB_post[
                        (n_local_i + sfb_smem_row_offset, k_chunk_idx_sfb)
                    ].to(self.acc_dtype)
                    chunk_acc_subtile[i] = chunk_acc_subtile[(i)] + sfb_i * acc_vec[(i)]

    @cute.jit
    def _chunked_close_publish_rowsum(
        self,
        warp_idx: cutlass.Int32,
        epi_tidx: cutlass.Int32,
        lane_rowsum_partial: cute.Tensor,
        s_b_rowsums: cute.Tensor,
        cta_tile_n: cutlass.Constexpr[int],
    ):
        """Per-SFA-chunk: tree-reduce lane_rowsum_partial -> s_b_rowsums and
        zero the partials for the next chunk. The caller must follow with
        epilog_sync_barrier.arrive_and_wait() to publish s_b_rowsums to all
        epilog warps before they read it in _chunked_close_apply.
        """
        _lane = epi_tidx % 32
        _n_per_warp: cutlass.Constexpr[int] = cta_tile_n // 4

        for _nw in cutlass.range_constexpr(_n_per_warp):
            _n_col = warp_idx * _n_per_warp + _nw
            _cs = lane_rowsum_partial[(_nw,)]
            for _off in cutlass.range_constexpr(5):
                _cs = _cs + cute.arch.shuffle_sync_down(_cs, 1 << _off)
            if _lane == 0:
                s_b_rowsums[(_n_col,)] = _cs
            lane_rowsum_partial[(_nw,)] = cutlass.Float32(0.0)

    @cute.jit
    def _chunked_close_apply(
        self,
        chunk_acc_sum: cute.Tensor,
        tTR_rAcc_final: cute.Tensor,
        tTR_rScale: cute.Tensor,
        s_b_rowsums: cute.Tensor,
        m_thr_offset: cute.Tensor,
    ):
        """Per-SFA-chunk: running += sfa*(512*chunk_acc_sum - 8*weighted_rowsum).

        Both ``chunk_acc_sum`` (already SFB-weighted, accumulated across
        the chunk's MMA tiles) and ``s_b_rowsums`` (likewise SFB-weighted,
        published by _chunked_close_publish_rowsum) are read here, so the
        per-chunk math collapses to: 1 SFA mul + 1 FMA per element. Resets
        chunk_acc_sum to 0 for the next chunk.
        """
        subtile_cnt = cute.size(tTR_rAcc_final.shape, mode=[3])

        for subtile_idx in cutlass.range_constexpr(subtile_cnt):
            chunk_acc_subtile = chunk_acc_sum[(None, None, None, subtile_idx)]
            running_subtile = tTR_rAcc_final[(None, None, None, subtile_idx)]
            scale_subtile = tTR_rScale[(None, None, None, subtile_idx)]
            scale = scale_subtile.load().to(self.acc_dtype)
            m_thr_sub_k = m_thr_offset[(None, None, None, subtile_idx)]
            for i in cutlass.range(
                cute.size(chunk_acc_subtile.shape), unroll_full=True
            ):
                n_local_i = m_thr_sub_k[(i)][1]
                weighted = s_b_rowsums[(n_local_i,)].to(self.acc_dtype)
                scale_i = scale[(i)]
                corrected_i = scale_i * (
                    512.0 * chunk_acc_subtile[(i)] - 8.0 * weighted
                )
                running_subtile[i] = running_subtile[(i)] + corrected_i
                chunk_acc_subtile[i] = cutlass.Float32(0.0)

    @cute.jit
    def _remap_sfb_flat_to_bsbc(
        self,
        sSFB_flat: cute.Tensor,
        sSFB_bsbc: cute.Tensor,
        tidx: cutlass.Int32,
        k_tile: cutlass.Int32,
    ):
        """Per-MMA-tile remap of fp32 SFB cache to E8M0 BSBC SMEM scratch."""
        cta_tile_n: cutlass.Constexpr[int] = self.cta_tile_shape_mnk[1]
        cta_tile_k: cutlass.Constexpr[int] = self.cta_tile_shape_mnk[2]
        num_sfb_slots_per_chunk: cutlass.Constexpr[int] = (
            self.sfb_granularity_k // self.sf_vec_size
        )
        num_sfb_chunks_per_mma_tile: cutlass.Constexpr[int] = max(
            cta_tile_k // self.sfb_granularity_k, 1
        )
        num_k_tiles_per_sfb_chunk: cutlass.Constexpr[int] = max(
            self.sfb_granularity_k // cta_tile_k, 1
        )
        num_sfb_slots_per_mma_tile: cutlass.Constexpr[int] = (
            cta_tile_k // self.sf_vec_size
        )
        total_cells: cutlass.Constexpr[int] = cta_tile_n * num_sfb_slots_per_mma_tile
        num_passes: cutlass.Constexpr[int] = (total_cells + 31) // 32

        lane = tidx % 32
        for _pass in cutlass.range_constexpr(num_passes):
            cell_idx = lane + _pass * 32
            if cell_idx < total_cells:
                n_local, k_slot = cute.idx2crd(
                    cell_idx, (cta_tile_n, num_sfb_slots_per_mma_tile)
                )
                _, k_chunk_in_tile = cute.idx2crd(
                    k_slot, (num_sfb_slots_per_chunk, num_sfb_chunks_per_mma_tile)
                )
                k_chunk_global = (
                    k_tile // num_k_tiles_per_sfb_chunk
                ) * num_sfb_chunks_per_mma_tile + k_chunk_in_tile
                val = sSFB_flat[(n_local, k_chunk_global)].to(self.sfb_dtype)
                n_inner, n_outer = cute.idx2crd(
                    n_local, (32, cute.ceil_div(cta_tile_n, 32))
                )
                if cutlass.const_expr(num_sfb_slots_per_mma_tile <= 4):
                    sSFB_bsbc[
                        (
                            ((n_inner, n_outer), (0, 0)),
                            0,
                            k_slot,
                            0,
                        )
                    ] = val
                else:
                    k_inner, k_outer = cute.idx2crd(
                        k_slot, (4, num_sfb_slots_per_mma_tile // 4)
                    )
                    sSFB_bsbc[
                        (
                            ((n_inner, n_outer), (0, 0)),
                            0,
                            (k_inner, k_outer),
                            0,
                        )
                    ] = val
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_warp()

    @staticmethod
    def _compute_stages_and_tmem_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: tuple[int, int, int],
        cta_tile_shape_mnk: tuple[int, int, int],
        epi_tile: cute.Tile,
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        sfa_dtype: type[cutlass.Numeric],
        sfb_dtype: type[cutlass.Numeric],
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        scale_granularity_m: int,
        sfa_granularity_k: int,
        sf_vec_size: int,
        smem_extra_bytes: int = 0,
        sfb_granularity_k: int = None,
        k_per_group: int = 0,
        is_mxscale_sfb: bool = False,
        num_acc_stage_override: Optional[int] = None,
    ) -> tuple[int, int, int, int, int, int, int, int, int]:
        """
        Compute pipeline stages and TMEM column allocation configurations.

        A is assumed TMEM-sourced (K-major A invariant). SMEM-source paths
        were removed.
        """
        # --- TMEM column budgets per stage ---
        # Accumulator D: 2 x for dual M-tile (tile-0 + tile-1), aligned to 2.
        acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
        tCtAcc_stage1 = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))
        num_tmem_acc_col_per_stage = cute.round_up(
            2 * tcgen05.find_tmem_tensor_col_offset(tCtAcc_stage1), 2
        )

        # Scale factors: SFA and SFB share the same per-stage TMEM footprint
        # under this layout. Formula: K / (sf_vec_size * 4), rounded up to 4.
        # When mma_K < BSBC atom K (sf_vec_size*4 = 128), pad to 128 so the
        # divide doesn't round down to 0 and the SFA unit-1.0 fill writes
        # to a valid TMEM region. Matches the round_up done in
        # _setup_attributes for the SFA BSBC SMEM layout.
        num_tmem_sf_col_per_stage = fp8_utils.blockscaled_scale_tmem_cols(
            cta_tile_shape_mnk[2], sf_vec_size
        )
        num_tmem_sfa_col_per_stage = num_tmem_sf_col_per_stage
        num_tmem_cols_sfb_per_stage = num_tmem_sf_col_per_stage

        # Converted A (TMEM-sourced): one 32-bit TMEM column holds 32 /
        # a_dtype.width elements. 2 x for dual M-tile, aligned to 4.
        num_a_elts_per_tmem_col = 32 // tiled_mma.op.a_dtype.width
        num_tmem_cols_a_per_stage = cute.round_up(
            2 * (cta_tile_shape_mnk[2] // num_a_elts_per_tmem_col),
            4,
        )

        # SFB single-stage SMEM layout - used below for b_scale_bytes_per_stage.
        sfb_smem_layout_one_stage = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, mma_tiler_mnk, sf_vec_size, 1
        )

        # SFA is filled once with unit 1.0 at prologue -> single-stage in TMEM.
        # SFB needs per-pipeline-stage TMEM cols for async-MMA / S2T-write
        # overlap (empirically required for perf).
        sm100_tmem_columns = cute.arch.get_max_tmem_alloc_cols("sm_100")
        max_tmem_cols_available = sm100_tmem_columns - num_tmem_sfa_col_per_stage

        # Per-pipeline-stage TMEM input cost: converted-A + SFB.
        ab_tmem_required_cols_per_stage = (
            num_tmem_cols_a_per_stage + num_tmem_cols_sfb_per_stage
        )

        # Heuristic for accumulator stage count:
        # Use as many stages as fit in the 512-column SM100 TMEM budget.
        # With A in TMEM we must leave room for the converted A buffer,
        # so we reduce acc stages based on accumulator column usage.
        # Caller may force a smaller value via ``num_acc_stage_override``
        # (used at large tiles e.g. prefill, where the heuristic-picked
        # 4 stages would exhaust TMEM and leave no room for trans2mma).
        if num_acc_stage_override is not None:
            accumulator_stage_count = num_acc_stage_override
        elif num_tmem_acc_col_per_stage <= 32:
            accumulator_stage_count = 5
        elif num_tmem_acc_col_per_stage <= 64:
            accumulator_stage_count = 4
        elif num_tmem_acc_col_per_stage < 128:
            accumulator_stage_count = 3
        elif num_tmem_acc_col_per_stage < 256:
            accumulator_stage_count = 2
        else:
            accumulator_stage_count = 1

        # Fixed overheads that are always reserved in SMEM:
        # bytes_per_pipeline_stage=16: each mbarrier pair costs 16 bytes.
        # tile_info_bytes: 2 stages x (4 Int32 + 16 mbar bytes).
        # c_bytes: C SMEM buffer (1 stage only for decode-style small N).
        # a_scale_bytes: 4 stages of scale SMEM buffer.
        # carveout_smem_bytes: sum of all fixed overheads to subtract from budget.
        bytes_per_pipeline_stage = 16
        # By default, we use 2 stages for tile info
        num_tile_info_stage = 2
        tile_info_bytes = (
            cute.size_in_bytes(cute.Int32, cute.make_layout((4, num_tile_info_stage)))
            + bytes_per_pipeline_stage * num_tile_info_stage
        )
        c_stage_count = 2
        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = int(c_bytes_per_stage * c_stage_count)

        smem_capacity = utils.get_smem_capacity_in_bytes("sm_100") - smem_extra_bytes

        # Ensure we have 4 buffers for scale tiles needed for 1 CTA tile
        a_scale_k_mode = max(cta_tile_shape_mnk[2] // sfa_granularity_k, 1)
        a_scale_m_mode = max(cta_tile_shape_mnk[0] // scale_granularity_m, 1)
        scale_load2accu_stage_count = 6
        a_scale_bytes_per_stage = fp8_utils.aligned_smem_bytes(
            sfa_dtype,
            cute.make_layout((a_scale_m_mode, a_scale_k_mode)),
        )
        a_scale_bytes = (
            a_scale_bytes_per_stage + bytes_per_pipeline_stage
        ) * scale_load2accu_stage_count

        sfb_extra_smem_bytes = 0
        if is_mxscale_sfb:
            sfb_bsbc_bytes = fp8_utils.aligned_smem_bytes(
                sfb_dtype,
                sfb_smem_layout_one_stage,
            )
            _sfb_g_eff = (
                sfb_granularity_k
                if sfb_granularity_k is not None
                else sfa_granularity_k
            )
            sfb_flat_bytes = fp8_utils.aligned_smem_bytes(
                sfb_dtype,
                cute.make_layout(
                    (cta_tile_shape_mnk[1], k_per_group // _sfb_g_eff),
                    stride=(1, cta_tile_shape_mnk[1]),
                ),
            )
            sfb_extra_smem_bytes = sfb_bsbc_bytes + sfb_flat_bytes

        carveout_smem_bytes = (
            bytes_per_pipeline_stage * accumulator_stage_count
            + a_scale_bytes
            + c_bytes
            + tile_info_bytes
            + sfb_extra_smem_bytes
        )

        # Total accumulator TMEM cols across all stages (4-aligned so the
        # converted-A region that follows also starts 4-aligned).
        num_tmem_acc_cols = cute.round_up(
            accumulator_stage_count * num_tmem_acc_col_per_stage, 4
        )

        # trans2mma stage count bounded by TMEM capacity (acc cols + input cols).
        transform2mma_stage_count_tmem_limited = (
            max_tmem_cols_available - num_tmem_acc_cols
        ) // ab_tmem_required_cols_per_stage
        if transform2mma_stage_count_tmem_limited <= 0:
            raise ValueError("Not enough TMEM capacity for selected tile size")

        # SMEM bytes per stage for raw A (pre-conversion) and for B.
        # 2 x A per stage - each stage carries two M-tiles (sA_tile0 + sA_tile1).
        # load2trans stages auto-reduce to fit the ~228 KB SMEM budget.
        a_load_bytes_per_stage = 2 * fp8_utils.aligned_smem_bytes(
            a_dtype,
            cute.make_layout((cta_tile_shape_mnk[0], cta_tile_shape_mnk[2])),
        )
        b_load_bytes_per_stage = fp8_utils.aligned_smem_bytes(
            b_dtype,
            cute.make_layout(
                (
                    cta_tile_shape_mnk[1] // cute.size(tiled_mma.thr_id),
                    cta_tile_shape_mnk[2],
                )
            ),
        )
        b_scale_bytes_per_stage = fp8_utils.aligned_smem_bytes(
            sfb_dtype,
            sfb_smem_layout_one_stage,
        )

        # Combined A+B load bytes per stage (including mbarrier overhead).
        ab_load_bytes_per_stage = int(
            a_load_bytes_per_stage
            + b_load_bytes_per_stage
            + b_scale_bytes_per_stage
            + 2 * bytes_per_pipeline_stage
        )

        # SMEM bytes per trans2mma stage. With A in TMEM there is no SMEM
        # for transformed A, only the mbarrier overhead.
        a_transform_bytes_per_stage = bytes_per_pipeline_stage

        # Potential trans2mma stage count bounded by SMEM capacity.
        transform2mma_stage_count_smem_limited = (
            smem_capacity - carveout_smem_bytes
        ) // (ab_load_bytes_per_stage + a_transform_bytes_per_stage)

        # Take the minimum of TMEM-limited and SMEM-limited stage counts.
        transform2mma_stage_count = min(
            transform2mma_stage_count_tmem_limited,
            transform2mma_stage_count_smem_limited,
        )

        # load2transform stage count: remaining SMEM after trans2mma stages.
        load2transform_stage_count = (
            smem_capacity
            - carveout_smem_bytes
            - (transform2mma_stage_count * a_transform_bytes_per_stage)
        ) // ab_load_bytes_per_stage

        # Sanity checks: must have at least 2 stages for each pipeline and
        # at least 1 accumulator stage to avoid deadlock.
        if (
            load2transform_stage_count < 2
            or transform2mma_stage_count < 2
            or accumulator_stage_count < 1
        ):
            raise ValueError(
                f"Not enough SMEM or TMEM capacity for selected tile size: {load2transform_stage_count=}, {transform2mma_stage_count=}, {accumulator_stage_count=}"
            )

        # Compute total TMEM columns for A (converted operand in TMEM path).
        num_tmem_a_cols = transform2mma_stage_count * num_tmem_cols_a_per_stage
        # SFB TMEM cols: per-pipeline-stage (empirically required for
        # async-MMA / S2T-write overlap).
        num_tmem_sfb_cols = transform2mma_stage_count * num_tmem_cols_sfb_per_stage

        # Try to use leftover SMEM for additional C output staging.
        # Check if we can increase c_stage_count with leftover smem
        c_stage_count += (
            smem_capacity
            - load2transform_stage_count * ab_load_bytes_per_stage
            - transform2mma_stage_count * a_transform_bytes_per_stage
            - scale_load2accu_stage_count * a_scale_bytes_per_stage
            - c_bytes
        ) // c_bytes_per_stage

        return (
            load2transform_stage_count,
            scale_load2accu_stage_count,
            transform2mma_stage_count,
            accumulator_stage_count,
            c_stage_count,
            num_tile_info_stage,
            num_tmem_acc_cols,
            num_tmem_a_cols,
            num_tmem_sfa_col_per_stage,
            num_tmem_sfb_cols,
        )

    @staticmethod
    def can_implement(
        mnkl: tuple[int, int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        c_dtype: type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
        sfa_granularity_k: int,
        mma_tiler: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
        sfb_granularity_k: int = None,
        sfb_dtype: type[cutlass.Numeric] = None,
    ) -> bool:
        """
        Check if the kernel can be implemented for the given tensor shapes and data types.
        """
        m, n, k, l = mnkl

        if not mixed_input_utils.is_valid_mma_tiler_and_cluster_shape(
            mma_tiler, cluster_shape_mn, False
        ):
            return False
        # Validate scale_granularity against whichever operand is int4. In
        # these MoE kernels, A is normally int4 weights and B is fp8
        # activations; the either-side check keeps this helper robust.
        quant_dtype = b_dtype if b_dtype == cutlass.Int4 else a_dtype
        if not mixed_input_utils.is_valid_scale_granularity(
            1, sfa_granularity_k, quant_dtype, k, mma_tiler[2]
        ):
            print(f"Invalid {sfa_granularity_k=} not a multiple of {mma_tiler[2]=}")
            return False
        sfb_ok, _, _ = fp8_utils.validate_sfb_policy(
            sfb_dtype,
            sfb_granularity_k,
            sfa_granularity_k,
            mma_tiler,
            k,
        )
        if not sfb_ok:
            return False

        if not (
            fp8_utils.check_contiguous_nb_alignment(
                a_dtype, m if a_major == "m" else k, 16
            )
            and fp8_utils.check_contiguous_nb_alignment(
                b_dtype, n if b_major == "n" else k, 16
            )
        ):
            return False
        return True


def auto_mma_tiler_mnk(
    sfa_granularity_k: int,
    sfb_granularity_k: int,
    sfb_dtype: type[cutlass.Numeric] = cutlass.BFloat16,
) -> tuple[int, int, int]:
    candidates = (
        (256, 128) if sfb_dtype == cutlass.Float8E8M0FNU else (256, 128, 64, 32)
    )
    for mma_k in candidates:
        if sfa_granularity_k % mma_k != 0:
            continue
        if sfb_dtype == cutlass.Float8E8M0FNU:
            if mma_k % sfb_granularity_k == 0 or sfb_granularity_k % mma_k == 0:
                return (128, 8, mma_k)
        elif sfb_granularity_k % mma_k == 0:
            return (128, 8, mma_k)
    raise ValueError(
        "Could not choose an MMA K tile; expected SFA/SFB granularities "
        "to support a decode tile in {32,64,128,256}."
    )


def run(
    m: int,
    g: int,
    k: int,
    n: int,
    sfa_granularity_k: int,
    a_dtype,
    b_dtype,
    c_dtype,
    acc_dtype,
    mma_tiler_mnk,
    cluster_shape_mn,
    tolerance: float,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    uniform_group_sizes: bool = True,
    use_cold_l2: bool = False,
    sfb_dtype=cutlass.BFloat16,
    sfb_granularity_k: int = None,
    num_acc_stage_override: Optional[int] = None,
):
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")

    # Pack (m, n, k, g) into mnkl - consumed as (M_out, N_tokens, K, G_experts).
    # 1-CTA only.
    if cluster_shape_mn[0] >= 2:
        raise ValueError(
            f"cluster_shape_mn[0] >= 2 (2-CTA) not supported, got cluster_shape_mn={cluster_shape_mn}"
        )
    # Resolve effective SFB granularity (defaults to SFA when omitted at the
    # runner level too), then use the same Tokamax-style K-tile default as
    # the CLI when imported callers pass mma_tiler_mnk=None.
    eff_sfb_granularity_k = (
        sfb_granularity_k if sfb_granularity_k is not None else sfa_granularity_k
    )
    if mma_tiler_mnk is None:
        mma_tiler_mnk = auto_mma_tiler_mnk(
            sfa_granularity_k, eff_sfb_granularity_k, sfb_dtype
        )
    mnkl = (m, n, k, g)
    ok = GroupedMixedInputGemmFp8.can_implement(
        mnkl,
        a_dtype,
        b_dtype,
        c_dtype,
        "k",  # a_major  (K-contiguous int4 weights)
        "k",  # b_major  (K-contiguous fp8 activations)
        "n",  # c_major  (token/N-contiguous output)
        sfa_granularity_k,
        mma_tiler_mnk,
        cluster_shape_mn,
        sfb_granularity_k=sfb_granularity_k,
        sfb_dtype=sfb_dtype,
    )
    if not ok:
        raise testing.CantImplementError("MoE GEMM configuration not supported")

    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    moe_kernel = GroupedMixedInputGemmFp8(
        sfa_granularity_k,
        acc_dtype,
        mma_tiler_mnk,
        cluster_shape_mn,
        g,
        k,
        sfb_granularity_k=eff_sfb_granularity_k,
        num_acc_stage_override=num_acc_stage_override,
    )

    torch.manual_seed(2025)
    tensors = create_tensors(
        m,
        g,
        k,
        n,
        a_dtype,
        b_dtype,
        c_dtype,
        sfa_granularity_k,
        uniform_group_sizes,
        sfb_dtype=sfb_dtype,
        sfb_granularity_k=eff_sfb_granularity_k,
    )
    assert tensors.a_scale is not None
    assert tensors.b_scale is not None

    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1],
    )

    compiled_kernel = cute.compile(
        moe_kernel,
        tensors.a.cute_tensor,
        tensors.a_scale.cute_tensor,
        tensors.b.cute_tensor,
        tensors.b_scale.cute_tensor,
        tensors.cumsum.cute_tensor,
        tensors.c.cute_tensor,
        max_active_clusters,
        current_stream,
    )
    print("COMPILE DONE")

    if not skip_ref_check:
        compiled_kernel(
            tensors.a.cute_tensor,
            tensors.a_scale.cute_tensor,
            tensors.b.cute_tensor,
            tensors.b_scale.cute_tensor,
            tensors.cumsum.cute_tensor,
            tensors.c.cute_tensor,
            current_stream,
        )
        torch.cuda.synchronize()
        run_ref_and_compare(
            tensors,
            c_dtype,
            tolerance,
        )
    elif iterations > 0:
        # CuTe helper preprocessing must run before testing.benchmark enters
        # its nested timing closure. This untimed launch mirrors the
        # reference-check path without doing host validation.
        compiled_kernel(
            tensors.a.cute_tensor,
            tensors.a_scale.cute_tensor,
            tensors.b.cute_tensor,
            tensors.b_scale.cute_tensor,
            tensors.cumsum.cute_tensor,
            tensors.c.cute_tensor,
            current_stream,
        )
        torch.cuda.synchronize()

    if iterations <= 0:
        return None

    def generate_tensors():
        tensors = create_tensors(
            m,
            g,
            k,
            n,
            a_dtype,
            b_dtype,
            c_dtype,
            sfa_granularity_k,
            uniform_group_sizes,
            sfb_dtype=sfb_dtype,
            sfb_granularity_k=eff_sfb_granularity_k,
        )
        assert tensors.a_scale is not None
        assert tensors.b_scale is not None
        return testing.JitArguments(
            tensors.a.cute_tensor,
            tensors.a_scale.cute_tensor,
            tensors.b.cute_tensor,
            tensors.b_scale.cute_tensor,
            tensors.cumsum.cute_tensor,
            tensors.c.cute_tensor,
            current_stream,
        )

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            tensors.a.ref_torch.numel() * tensors.a.ref_torch.element_size()
            + tensors.b.ref_torch.numel() * tensors.b.ref_torch.element_size()
            + tensors.c.ref_torch.numel() * tensors.c.ref_torch.element_size()
            + tensors.b_scale.ref_torch.numel()
            * tensors.b_scale.ref_torch.element_size()
        )
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = testing.benchmark(
        compiled_kernel,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )
    return exec_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Contiguous grouped fp8 mixed-input GEMM runner."
    )
    # Grouped/MoE problem shape - MxNxKxG where:
    #   M = per-expert output dim (= weights' outer dim; kernel MMA-M axis)
    #   N = total tokens          (= activations' outer dim; kernel MMA-N axis)
    #   K = hidden dim (reduction)
    #   G = number of experts
    # This matches CUTLASS GEMM convention where A (LHS) is the M_out x K
    # operand (here: weights) and B (RHS) is the N_tokens x K operand
    # (here: activations). If
    # you're used to ML naming where M=tokens, N=output_dim, just flip.
    parser.add_argument(
        "--m", type=int, default=3072, help="Per-expert output dim (M_out)"
    )
    parser.add_argument("--g", type=int, default=256, help="Number of experts (groups)")
    parser.add_argument(
        "--k", type=int, default=2048, help="Hidden dim (reduction axis)"
    )
    parser.add_argument("--n", type=int, default=512, help="Total tokens (N_tokens)")

    parser.add_argument(
        "--mma_tiler_mnk",
        type=fp8_utils.parse_comma_separated_ints,
        default=None,
        help=(
            "Kernel tile shape (M_kernel, N_kernel, K_tile). If omitted, "
            "selects 128,8,K. BF16 SFB may use K down to 32; E8M0 HW SFB "
            "uses the largest supported K >=128 co-aligned with SFB."
        ),
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=fp8_utils.parse_comma_separated_ints,
        default=(1, 1),
    )

    parser.add_argument(
        "--a_dtype",
        type=cutlass.dtype,
        default=cutlass.Int4,
        choices=[cutlass.Int4],
        help="A operand (MMA LHS) dtype - int4 weights",
    )
    parser.add_argument(
        "--b_dtype",
        type=cutlass.dtype,
        default=cutlass.Float8E4M3FN,
        choices=[cutlass.Float8E4M3FN, cutlass.Float8E5M2],
        help="B operand (MMA RHS) dtype - fp8 activations",
    )
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--acc_dtype", type=cutlass.dtype, default=cutlass.Float32)
    parser.add_argument(
        "--sfb_dtype",
        type=cutlass.dtype,
        default=cutlass.BFloat16,
        choices=[cutlass.BFloat16, cutlass.Float8E8M0FNU],
        help=(
            "SFB (activation scale) dtype. BFloat16 uses software SFB in the "
            "epilogue; Float8E8M0FNU uses the MX hardware SFB path."
        ),
    )

    parser.add_argument("--sfa_granularity_k", type=int, default=256)
    parser.add_argument(
        "--sfb_granularity_k",
        type=int,
        default=None,
        help=(
            "K-elements per SFB (activation) scale. Defaults to "
            "sfa_granularity_k when omitted. Must divide K. BF16 SFB requires "
            "a multiple of mma_tiler_mnk[2]; E8M0 SFB allows a multiple or "
            "divisor of mma_tiler_mnk[2] and requires a multiple of 32. "
            "Smaller values give finer-grained activation quantization."
        ),
    )
    parser.add_argument("--tolerance", type=float, default=1e-01)
    parser.add_argument("--warmup_iterations", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument(
        "--non_uniform_group_sizes",
        action="store_true",
        help="If set, use random group sizes instead of uniform M/G.",
    )
    parser.add_argument("--use_cold_l2", action="store_true")
    parser.add_argument(
        "--num_acc_stage",
        type=int,
        default=None,
        help=(
            "Override the auto-tuned accumulator-pipeline stage count. "
            "Lower = less acc TMEM but less acc/MMA overlap. Useful for "
            "fitting larger MMA tiles (e.g. prefill at N=32+)."
        ),
    )
    args = parser.parse_args()
    eff_sfb_granularity_k = (
        args.sfb_granularity_k
        if args.sfb_granularity_k is not None
        else args.sfa_granularity_k
    )
    if args.mma_tiler_mnk is None:
        try:
            args.mma_tiler_mnk = auto_mma_tiler_mnk(
                args.sfa_granularity_k, eff_sfb_granularity_k, args.sfb_dtype
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(f"auto_mma_tiler_mnk={args.mma_tiler_mnk}")
    print(f"skip_ref_check={args.skip_ref_check}")
    print(f"MoE problem: M_out={args.m} G={args.g} K={args.k} N_tokens={args.n}")

    exec_time = run(
        args.m,
        args.g,
        args.k,
        args.n,
        args.sfa_granularity_k,
        args.a_dtype,
        args.b_dtype,
        args.c_dtype,
        args.acc_dtype,
        args.mma_tiler_mnk,
        args.cluster_shape_mn,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        not args.non_uniform_group_sizes,
        use_cold_l2=args.use_cold_l2,
        sfb_dtype=args.sfb_dtype,
        sfb_granularity_k=args.sfb_granularity_k,
        num_acc_stage_override=args.num_acc_stage,
    )
    print("PASS")
    print(f"{exec_time=}")
