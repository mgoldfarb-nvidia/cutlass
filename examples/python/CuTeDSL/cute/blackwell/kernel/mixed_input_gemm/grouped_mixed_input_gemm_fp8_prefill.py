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

"""FP8 grouped GEMM prefill with dtype-selected SFB policy.

This file hosts the prefill variant. It has the same operand roles as
``grouped_mixed_input_gemm_fp8.py`` but moves the B-rowsum computation into a
small device precompute kernel that runs immediately before the main GEMM. The
main GEMM then stages one fp32 rowsum value per ``(token, mma_K tile)``.

The SFB policy is selected at compile time from ``sfb.element_type``:
``BFloat16`` uses post-MMA software SFB, while ``Float8E8M0FNU`` uses the
MX/HW-SFB path.

The example implements a contiguous grouped/MoE GEMM. The weights are grouped
by expert while the activations are one contiguous token buffer partitioned by
``cumsum``:

  * A / ``a``: int4 weights, ``(M_out, K, G)``, transformed to fp8 in-kernel.
  * SFA / ``sfa``: bf16 weight scales, ``(M_out, K / sfa_g, G)``.
  * B / ``b``: fp8 activations, ``(N_tokens, K, 1)``, consumed directly by MMA.
  * SFB / ``sfb``: bf16 or E8M0 activation scales,
    ``(N_tokens, K / sfb_g, 1)``.
  * rowsum: fp32 workspace, ``(N_tokens, K / mma_K, 1)``. For bf16 SFB it
    stores unweighted B rowsums; for E8M0 SFB it stores SFB-weighted rowsums.
  * C / ``c``: bf16 output in user layout ``(N_tokens, M_out, 1)``.

The rowsum workspace is intentionally not part of the host tensor-creation
helper; it is a per-kernel implementation detail and is allocated by the
runner before compilation or benchmarking.

To run this example from ``examples/python/CuTeDSL``:

.. code-block:: bash

    python cute/blackwell/kernel/mixed_input_gemm/grouped_mixed_input_gemm_fp8_prefill.py \
      --m 3072 --g 256 --k 2048 --n 512                                                                 \
      --mma_tiler_mnk 128,8,256 --sfa_granularity_k 256 --sfb_granularity_k 256                         \
      --a_dtype Int4 --b_dtype Float8E4M3FN --sfb_dtype BFloat16                                        \
      --c_dtype BFloat16 --acc_dtype Float32

Use ``--use_pdl_rowsum`` to connect the rowsum precompute and main GEMM with
Programmatic Dependent Launch.

To collect performance with NCU:

.. code-block:: bash

    ncu --target-processes all                                                                           \
      python cute/blackwell/kernel/mixed_input_gemm/grouped_mixed_input_gemm_fp8_prefill.py \
      --m 3072 --g 256 --k 2048 --n 512                                                                 \
      --mma_tiler_mnk 128,8,256 --sfa_granularity_k 256 --sfb_granularity_k 256                         \
      --warmup_iterations 1 --iterations 10 --skip_ref_check --use_pdl_rowsum
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
    build_rowsum_tensor,
    create_tensors_for_contiguous_grouped_mixed_input_gemm_fp8 as create_tensors,
    run_ref_and_compare_contiguous_grouped_mixed_input_gemm_fp8 as run_ref_and_compare,
)
from blackwell.kernel.mixed_input_gemm import (
    grouped_mixed_input_gemm_fp8_utils as fp8_utils,
)


class GroupedMixedInputGemmFp8Prefill:
    """Contiguous grouped fp8 GEMM with dtype-selected SFB policy for SM100.

    Operands (CUTLASS naming - A is the quantized LHS, regardless of ML role):
      * ``a``     : int4 weights, shape (M_out, K, G). MMA LHS; converted to
        fp8 inline by the TRANSFORM warps using the biased ``int4 + 8`` ->
        fp8 subnormal trick. Final-output correction is applied in the epilog.
      * ``sfa``   : bf16 per-(M_out x k_group x expert) weight scales. Real
        per-M-out scales applied in the epilog (the TMEM SFA channel is
        filled with unit 1.0 since bf16 doesn't fit through E8M0/E2M1).
      * ``b``     : fp8 activations, shape (N_tokens, K, 1). MMA RHS; fed
        directly into the TCU.
      * ``sfb``   : per-(token x k_group) activation scales in the regular
        producer layout ``(N_tokens, K/sfb_g, 1)``. ``BFloat16`` selects the
        post-MMA software SFB path; ``Float8E8M0FNU`` selects the HW-SFB path
        that remaps the regular layout into BSBC SMEM before the tcgen05 S2T
        copy.
      * ``cumsum``: (G+1,) int32 expert-boundary offsets along the token axis,
        used by the persistent scheduler for ragged-dot dispatch.
      * ``c``     : bf16 output, shape (N_tokens, M_out, 1).

    Related kernels:
      * ``grouped_mixed_input_gemm_fp8.py`` - dtype-selected decode path
        without the rowsum precompute; the epilog computes B rowsums from
        staged B.

    This prefill variant handles independent SFA/SFB granularities. It accepts
    regular SFB input from previous kernels and, when the dtype is E8M0,
    applies the hardware block layout inside this kernel.

    Architectural features:
      * **Single TRANSFORM warp group** - one 4-warp TRANSFORM group processes
        the local M tile for each k_tile. 12 warps total: 4 epilog + TMA +
        MMA + scale_TMA + scheduler + 4 transform.
      * **Single local A-TMA per CTA** - each stage carries one local M tile.
        In 2-CTA mode, CtaGroup/multicast accounting supplies the peer-CTA
        participation.
      * **2-CTA handoff support** - in ``cluster_shape_mn=(2,1)`` the two
        CTAs cooperatively compute one MMA-M tile; each CTA owns half the M
        rows while sharing the B tile and, for E8M0 SFB, the HW scale source.
      * **Unified acc / scale_load2accu pipelines** - one acc TMEM region and
        one SFA scale buffer are tracked per local CTA tile; in 2-CTA mode
        the barriers include the peer CTA.
      * **LOP3-fused biased int4->fp8 conversion** - the ``(src ^ 0x88) & 0x0F``
        and ``(src >> 4) ^ 0x08) & 0x0F`` patterns are emitted as single
        ``lop3.b32`` instructions (LUT 0x28).
      * **per-MMA-tile MMA via accumulate=False + running f32 acc** - each
        MMA tile uses ``accumulate=False`` and produces a fresh per-tile s32
        acc. This enables independent sfa/sfb granularity; E8M0 SFB is
        supplied through the HW blockscaled MMA scale channel, while bf16 SFB
        and SFA remain post-MMA epilog scales.
      * **chunked-rescale epilog** (key perf optimization at small mma_K) -
        when several MMA K tiles share one SFA tile, the epilog accumulates
        per-MMA-tile corrected partial sums in RMEM and applies SFA once at
        the SFA-chunk boundary. When ``sfa_g == mma_K`` this collapses to the
        fused per-tile fast path.

    Parameters:
      sfa_granularity_k : K elements per weight-scale; must be a multiple
                            of ``mma_tiler_mnk[2]``.
      sfb_granularity_k : K elements per activation-scale. For bf16 SFB it
                            must be a multiple of ``mma_tiler_mnk[2]``; for
                            E8M0 SFB it may be a multiple or divisor of
                            ``mma_tiler_mnk[2]`` and must be a multiple of
                            32. Defaults to ``sfa_granularity_k``.
      acc_dtype           : accumulator dtype (typically ``cutlass.Float32``).
      mma_tiler_mnk       : (MMA-M, MMA-N, MMA-K) tile shape. MMA-K must be
                            co-aligned with ``sfb_granularity_k``; the E8M0
                            HW-SFB policy additionally requires ``mma_K >=
                            128`` for the BSBC scale atom.
      cluster_shape_mn    : (M, N) cluster shape. Supports (1, 1) and 2-CTA
                            handoff via (2, 1).
      group_count         : number of experts (G).
      k_per_group         : full K dimension; saved as ``self.k_total`` for
                            constexpr use.

    Known limitations:
      * ``mma_tiler_K < 128`` is rejected for the MX HW-SFB path because the
        BSBC scale atom covers 128 K elements. ``sfb_granularity_k=32`` is
        still supported when used with a larger MMA-K tile such as the
        production ``mma_tiler_K=256`` path.
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
        use_simt_store: bool = False,
        use_pdl_rowsum: bool = False,
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
        self.use_simt_store = use_simt_store
        self.use_pdl_rowsum = use_pdl_rowsum
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
        # Resolved in __call__ from sfb.element_type. Keep a conservative
        # default so constructor-time shape checks can admit both policies.
        self.is_mxscale_sfb = False

        if cutlass.const_expr(self.sfa_granularity_k % mma_tiler_mnk[2] != 0):
            raise ValueError(
                "sfa_granularity_k must be exactly multiple of CTA tile shape K"
            )
        # sfb_granularity_k can be either a multiple OR a divisor of mma_K.
        # - sfb_g >= mma_K: sfb constant within a K-tile; rowsum granularity
        #   stays at mma_K and the per-K-tile correction uses the single
        #   sfb*rowsum value.
        # - sfb_g < mma_K (e.g. sfb=32 mxfp8 native, mma_K=256): the K-tile
        #   spans (mma_K/sfb_g) sub-blocks each with its own sfb. Rowsum
        #   granularity drops to sfb_g (sub-rowsum), and the correction sums
        #   sfb_sub * rowsum_sub over the sub-blocks.
        if cutlass.const_expr(
            (self.sfb_granularity_k >= mma_tiler_mnk[2])
            and (self.sfb_granularity_k % mma_tiler_mnk[2] != 0)
        ):
            raise ValueError(
                "sfb_granularity_k must be a multiple OR a divisor of mma_K"
            )
        if cutlass.const_expr(
            (self.sfb_granularity_k < mma_tiler_mnk[2])
            and (mma_tiler_mnk[2] % self.sfb_granularity_k != 0)
        ):
            raise ValueError("sfb_granularity_k must divide mma_K when sfb_g < mma_K")
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
        self.sfb_smem_ready_barrier = pipeline.NamedBarrier(
            3, 32 * (len(self.epilog_warp_id) + 1)
        )
        self.sched_sync_barrier = pipeline.NamedBarrier(4, 32)

        self.smem_buffer_align_bytes = 1024
        self.sfb_tmem_margin_cols = 16

    def _setup_attributes(self):
        """
        Set up configurations that are dependent on GEMM inputs.

        A is always assumed K-major, so the transformed-A destination is
        always TMEM (SMEM path is dead).
        """
        # 2-CTA mode is selected via cluster_shape_mn[0] == 2; the cluster
        # cooperatively computes one (mma_tiler_M x mma_tiler_N) MMA tile,
        # so each CTA owns mma_tiler_M / 2 rows.
        self.use_2cta_instrs = self.cluster_shape_mn[0] == 2
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        if cutlass.const_expr(self.is_mxscale_sfb):
            # Block-scaled MMA (`tcgen05.mma.kind::mxf8f6f4`): HW applies SFB
            # along the K axis at sf_vec_size=32 grain. SFA TMEM channel is
            # filled with E8M0 unit 1.0 at prologue (post-MMA SFA application
            # via running f32 accumulator); SFB TMEM is written per-MMA-tile
            # via tcgen05.cp from BSBC SMEM scratch.
            self.tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                cutlass.Float8E8M0FNU,
                self.sf_vec_size,
                self.cta_group,
                self.mma_tiler[:2],
                tcgen05.OperandSource.TMEM,
            )
        else:
            # BF16 SFB is applied in the epilog, so the MMA itself is the
            # regular fp8 tcgen05 path and both hardware scale channels are
            # identity.
            self.tiled_mma = sm100_utils.make_trivial_tiled_mma(
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.acc_dtype,
                self.cta_group,
                self.mma_tiler[:2],
                tcgen05.OperandSource.TMEM,
            )

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // (2 if self.use_2cta_instrs else 1),
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

        # 2-CTA aware multicast accounting. cluster_layout_vmnk has shape
        # (V, M, N, K). num_mcast_ctas_a counts CTAs along N (B-side multicast
        # for A); num_mcast_ctas_b counts CTAs along M (A-side multicast for B).
        # In cluster (2,1) with 2-CTA both are 1, but the masks must still be
        # built when use_2cta_instrs is on so the cp.async.bulk.tensor TMA
        # carries the cta_group::2 peer-arrive accounting that satisfies the
        # transaction-mbarrier expect_tx (= 2 * b_copy_size).
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

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
            self.k_total,
            self.epi_tile,
            self.a_dtype,
            self.b_dtype,
            self.sfa_dtype,
            self.sfb_dtype,
            self.rowsum_dtype,
            self.c_dtype,
            self.c_layout,
            self.scale_granularity_m,
            self.sfa_granularity_k,
            self.sfb_granularity_k,
            self.sf_vec_size,
            self.is_mxscale_sfb,
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

        # Get smem layout for SFA (used to help with tmem layout). The BF16
        # post-SFB path allows mma_K < the 128-element BSBC atom extent, so pad
        # the layout used for identity scale fills in that policy.
        _sfa_layout_mma_tiler = self.mma_tiler
        if cutlass.const_expr(not self.is_mxscale_sfb):
            _sfa_layout_mma_tiler = (
                self.mma_tiler[0],
                self.mma_tiler[1],
                cute.round_up(self.mma_tiler[2], self.bsbc_min_k),
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
            # SFB SMEM uses BSBC (Block-Scaled Basic Chunk) layout for HW SFB
            # via `tcgen05.cp` -> TMEM SFB -> `tcgen05.mma.kind::mxf8f6f4`.
            # Epilog warps stage regular SFB GMEM directly into per-K BSBC
            # stages. In 2-CTA mode this gives both CTAs a populated local SFB
            # source before the leader issues the cooperative S2T copy.
            self.sfb_bsbc_stage_count = self.k_total // self.cta_tile_shape_mnk[2]
            self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
                self.tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                self.sfb_bsbc_stage_count,
            )
            self.sfb_smem_layout_per_stage = cute.slice_(
                self.sfb_smem_layout_staged, (None, None, None, 0)
            )
        else:
            self.sfb_bsbc_stage_count = 1
            self.sfb_smem_layout_staged = self.sfa_smem_layout_staged
            self.sfb_smem_layout_per_stage = self.sfa_smem_layout_per_stage

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        sfa: cute.Tensor,
        b: cute.Tensor,
        sfb: cute.Tensor,
        rowsum: cute.Tensor,
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
          ``sfb``    : activation scales in regular
                       (N_tokens, K / sfb_granularity_k, 1) layout. BFloat16
                       is applied post-MMA; E8M0 is remapped into BSBC SMEM
                       before HW SFB.
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
        # SFB arrives in the regular producer layout (N_tokens, K/sfb_g, 1).
        # Its dtype selects the compile-time policy:
        #   * BFloat16: post-MMA software SFB in the epilog.
        #   * E8M0: HW SFB through BSBC SMEM -> TMEM -> blockscaled MMA.
        self.sfb_dtype: type[cutlass.Numeric] = sfb.element_type
        self.is_mxscale_sfb = self.sfb_dtype == cutlass.Float8E8M0FNU
        # Device-filled B rowsum workspace, one fp32 value per (token, k_tile).
        # BF16 stores unweighted B rowsums and applies SFB in the epilog; E8M0
        # stores SFB-weighted rowsums that match the HW-SFB MMA accumulation.
        # Launching the rowsum precompute before this GEMM avoids an in-GEMM
        # cooperative B-drain.
        self.rowsum_dtype: type[cutlass.Numeric] = rowsum.element_type
        # The HW scale channels that tcgen05.mma reads are E8M0. SFA is filled
        # with identity because bf16 SFA is applied in the epilog. The E8M0
        # policy copies SFB through BSBC SMEM; BF16 SFB stays in the epilog.
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
                f"Unsupported SFB dtype {self.sfb_dtype}; expected BFloat16 or Float8E8M0FNU"
            )
        if cutlass.const_expr(self.is_mxscale_sfb):
            if cutlass.const_expr(self.mma_tiler[2] < self.bsbc_min_k):
                raise ValueError(
                    f"mxscale HW-SFB requires mma_tiler_K >= {self.bsbc_min_k}"
                )
            if cutlass.const_expr(self.sfb_granularity_k % self.sf_vec_size != 0):
                raise ValueError(
                    f"sfb_granularity_k must be a multiple of sf_vec_size={self.sf_vec_size}"
                )
        else:
            if cutlass.const_expr(self.sfb_granularity_k % self.mma_tiler[2] != 0):
                raise ValueError(
                    "BF16 SFB requires sfb_granularity_k to be a multiple of mma_tiler_K"
                )

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

        # Set up gmem copy atoms for A. In 2-CTA mode A's TMA is also
        # CtaGroup.TWO so the bulk-tensor copy participates in the cluster's
        # 2-CTA accounting (mirrors blockwise_gemm.py reference).
        a_op = mixed_input_utils.get_tma_atom_kind(
            self.is_a_mcast, self.use_2cta_instrs, is_b=False
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Set up gmem copy atoms for B. In 2-CTA mode the TMA atom is
        # CtaGroup.TWO so the bulk-tensor copy delivers B to both CTAs in
        # the cluster's M-axis (the 2-CTA tcgen05.mma consumes a shared B).
        # No explicit M-axis multicast (mcast=False) - the 2-CTA TMA itself
        # broadcasts to the CTA pair.
        b_op = mixed_input_utils.get_tma_atom_kind(
            False, self.use_2cta_instrs, is_b=True
        )
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

        # Rowsum is loaded into sRowsum_post via cp.async.
        # Native shape (N_tokens, K/mma_K, L=1) fp32 - one rowsum value per
        # (token, K-tile), where K-tile = mma_K = cta_tile_K. BF16 stores
        # unweighted B rowsums; E8M0 stores SFB-weighted rowsums.
        # Indexed as rowsum[(n_global, k_tile_idx, 0)] in the cp.async loop.
        tma_tensor_rowsum = rowsum

        # Calculate copy size for tensor A, B, scale
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        scale_copy_size = cute.size_in_bytes(self.sfa_dtype, scale_smem_layout)

        self.num_tma_load_bytes_a = a_copy_size
        # load2mma_pipeline carries B only. SFB is regular GMEM staged by
        # epilog warps directly into BSBC SMEM in-kernel.
        self.num_tma_load_bytes_b = b_copy_size * cute.size(self.tiled_mma.thr_id.shape)
        # Scale pipeline carries one local SFA tile per CTA; 2-CTA accounting
        # is handled by the CtaGroup/multicast TMA path.
        self.num_tma_load_bytes_scale = scale_copy_size
        self.tile_sched_params, grid = fp8_utils.compute_persistent_grid(
            c,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
            m_tile_multiplier=1,
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

            # load2trans - A TMA -> SMEM for the single local CTA tile.
            a_load2trans_full_mbar_ptr: _MemRange[_Int64, _n_load2trans]
            a_load2trans_empty_mbar_ptr: _MemRange[_Int64, _n_load2trans]

            # scale_load2accu - SFA TMA -> epilog for the single local CTA
            # tile. In 2-CTA mode, CtaGroup/multicast accounting handles
            # the cluster peer participation.
            a_scale_load2accu_full_mbar_ptr: _MemRange[_Int64, _n_scale]
            a_scale_load2accu_empty_mbar_ptr: _MemRange[_Int64, _n_scale]

            # trans2mma - TRANSFORM warps -> MMA for the local CTA tile.
            # Transform writes TMEM, then arrives once; MMA waits once,
            # issues tcgen05.mma, and releases once.
            a_trans2mma_full_mbar_ptr: _MemRange[_Int64, _n_trans2mma]
            a_trans2mma_empty_mbar_ptr: _MemRange[_Int64, _n_trans2mma]

            # b_load2mma - B TMA -> MMA (shares num_load2trans_stage depth with A).
            b_load2mma_full_mbar_ptr: _MemRange[_Int64, _n_load2trans]
            b_load2mma_empty_mbar_ptr: _MemRange[_Int64, _n_load2trans]

            # acc - MMA -> epilog for the local CTA tile. MMA commits once per
            # k-block; epilog waits once and consumes the acc TMEM region
            # before releasing.
            acc_full_mbar_ptr: _MemRange[_Int64, _n_acc]
            acc_empty_mbar_ptr: _MemRange[_Int64, _n_acc]

            # Cross-CTA SFB readiness handshake. In 2-CTA mode SFB is staged by
            # epilog warps, not a cluster-aware TMA pipeline, so the leader MMA
            # warp needs an explicit peer-ready signal before tcgen05.cp.cta_group::2.
            sfb_cluster_mbar: _Int64

            # TMEM allocator handshake.
            tmem_dealloc_mbar: _Int64
            tmem_holding_buf: cutlass.Int32

        self.shared_storage = SharedStorage

        # ---- Launch rowsum precompute first (writes into the rowsum
        # workspace; same stream so downstream GEMM observes the writes).
        # Rowsum granularity is ALWAYS mma_K (one cell per token per K-tile).
        # BF16 rowsum is unweighted. E8M0 rowsum folds in SFB weighting
        # internally - at sfb_g < mma_K it sums (mma_K / sfb_g) sub-blocks
        # each weighted by its SFB. GEMM epilog then does one rowsum read per
        # element per K-tile.
        # Grid: (num_K_tiles, ceil(N_tokens / N_PER_BLOCK), 1).
        _ROWSUM_N_PER_BLOCK: cutlass.Constexpr[int] = 128
        _rowsum_n_total = b.shape[0]
        _rowsum_chunks_const: cutlass.Constexpr[int] = (
            self.k_total // self.cta_tile_shape_mnk[2]
        )
        _rowsum_n_blocks = (
            _rowsum_n_total + _ROWSUM_N_PER_BLOCK - 1
        ) // _ROWSUM_N_PER_BLOCK
        self._rowsum_kernel(
            b,
            sfb,
            tma_tensor_rowsum,
            self.cta_tile_shape_mnk[2],  # mma_K
            self.sfb_granularity_k,
            _ROWSUM_N_PER_BLOCK,
        ).launch(
            grid=(_rowsum_chunks_const, _rowsum_n_blocks, 1),
            block=[_ROWSUM_N_PER_BLOCK, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl_rowsum,
        )

        self.kernel(
            self.tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_scale,
            tma_tensor_scale,
            tma_atom_b,
            tma_tensor_b,
            sfb,
            tma_tensor_rowsum,
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
            use_pdl=self.use_pdl_rowsum,
        )
        return

    # GPU device kernel: B-rowsum precompute. Output is one fp32 cell per
    # (token, K_tile) where K_tile = cta_K = mma_K. BF16 stores the unweighted
    # B rowsum. E8M0 stores:
    # ``sum_{sub in K_tile} sfb[token, sub] * sum_{k in sub} B[token, k]``,
    # the SFB-weighted sub-rowsum summed over all (mma_K / sfb_g) sub-blocks
    # within the K-tile.
    #
    # For E8M0, this hoists the sfb*rowsum work from the GEMM epilog into the
    # precompute. BF16 keeps SFB application in the epilog.
    #
    # When sfb_g >= mma_K the inner sub loop has 1 iteration and weighted_acc
    # = sfb * sum_k B (= unweighted rowsum x sfb).
    @cute.kernel
    def _rowsum_kernel(
        self,
        mB_nkl: cute.Tensor,
        mSFB_nkl: cute.Tensor,
        mRowsum_nkl: cute.Tensor,
        cta_K: cutlass.Constexpr[int],
        sfb_g: cutlass.Constexpr[int],
        n_per_block: cutlass.Constexpr[int],
    ):
        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(self.use_pdl_rowsum):
            cute.arch.griddepcontrol_launch_dependents()
        chunk_idx = bidx  # K-tile index in [0, K_total/cta_K)
        token_block_start = bidy * n_per_block
        n_total = mB_nkl.shape[0]

        # Sub-block parameters (constexpr).
        # num_subs = number of sfb sub-blocks per K-tile.
        # num_k_tiles_per_sfb_chunk = number of K-tiles that share one sfb
        # value (only > 1 when sfb_g > mma_K, e.g. sfb=512 / mma_K=256 -> 2).
        num_subs: cutlass.Constexpr[int] = max(cta_K // sfb_g, 1)
        num_k_tiles_per_sfb_chunk: cutlass.Constexpr[int] = max(sfb_g // cta_K, 1)
        elems_per_sub: cutlass.Constexpr[int] = cta_K // num_subs  # = min(sfb_g, cta_K)

        # SMEM tile (n_per_block x cta_K) fp8; use cp.async via int32 recast so
        # each issued instruction transfers 4 fp8 bytes. Pad each row by 4
        # fp8 elements so the per-token row reductions don't read all lanes
        # from the same SMEM bank when cta_K is a multiple of 128 bytes.
        _sB_stride: cutlass.Constexpr[int] = cta_K + 4
        smem = utils.SmemAllocator()
        sB = smem.allocate_tensor(
            element_type=mB_nkl.element_type,
            layout=cute.make_layout((n_per_block, cta_K), stride=(_sB_stride, 1)),
            byte_alignment=16,
        )

        _atom_cpasync = cute.make_copy_atom(
            cpasync.CopyG2SOp(), cutlass.Int32, num_bits_per_copy=32
        )
        # Each int32 = 4 fp8 bytes packed along K. The block needs
        # n_per_block x cta_K / 4 int32 cells.
        _total_int32: cutlass.Constexpr[int] = (n_per_block * cta_K) // 4
        _per_thr_loads: cutlass.Constexpr[int] = (
            _total_int32 + n_per_block - 1
        ) // n_per_block

        mB_i32 = cute.recast_tensor(mB_nkl, cutlass.Int32)
        sB_i32 = cute.recast_tensor(sB, cutlass.Int32)
        # K-int32 stride per token: cta_K // 4 (the row in sB).
        _k_i32_per_row: cutlass.Constexpr[int] = cta_K // 4
        for li in cutlass.range_constexpr(_per_thr_loads):
            _idx = tidx + li * n_per_block
            if _idx < _total_int32:
                _row = _idx // _k_i32_per_row
                _col = _idx % _k_i32_per_row
                _g_token = token_block_start + _row
                _g_k_i32 = chunk_idx * _k_i32_per_row + _col
                if _g_token < n_total:
                    _src = fp8_utils.make_single_element_tensor_view(
                        mB_i32, (_g_token, _g_k_i32, 0)
                    )
                    _dst = fp8_utils.make_single_element_tensor_view(
                        sB_i32, (_row, _col)
                    )
                    cute.copy(_atom_cpasync, _src, _dst)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # Reduce: each thread handles one (token_local, K_tile) output. The
        # BF16 policy writes the unweighted B rowsum. The MX policy writes the
        # SFB-weighted rowsum matching the hardware-scaled MMA accumulation.
        #
        # sfb_chunk_global indexing:
        # - sfb_g < cta_K (sfb=32/mma_K=256, num_subs=8): each sub has its
        #   own sfb_chunk_global = chunk_idx * num_subs + sub.
        # - sfb_g >= cta_K (sfb=256/mma_K=256, num_subs=1; or sfb=512/mma_K=256,
        #   num_subs=1 num_k_tiles_per_sfb_chunk=2): single sub, sfb_chunk_global
        #   = chunk_idx // num_k_tiles_per_sfb_chunk.
        if tidx < n_per_block:
            _g_token = token_block_start + tidx
            if _g_token < n_total:
                if cutlass.const_expr(self.is_mxscale_sfb):
                    weighted_acc = cutlass.Float32(0.0)
                    for sub in cutlass.range_constexpr(num_subs):
                        if cutlass.const_expr(num_subs > 1):
                            # sfb_g < cta_K: per-sub sfb chunk
                            _sfb_chunk_global = chunk_idx * num_subs + sub
                        else:
                            # sfb_g >= cta_K: shared sfb chunk across K-tiles
                            _sfb_chunk_global = chunk_idx // num_k_tiles_per_sfb_chunk
                        sfb_sub = mSFB_nkl[_g_token, _sfb_chunk_global, 0].to(
                            cutlass.Float32
                        )
                        sub_acc = cutlass.Float32(0.0)
                        for k_off in cutlass.range_constexpr(elems_per_sub):
                            v = sB[tidx, sub * elems_per_sub + k_off]
                            sub_acc = sub_acc + v.to(cutlass.Float32)
                        weighted_acc = weighted_acc + sfb_sub * sub_acc
                    mRowsum_nkl[_g_token, chunk_idx, 0] = weighted_acc
                else:
                    acc = cutlass.Float32(0.0)
                    for k_off in cutlass.range_constexpr(cta_K):
                        v = sB[tidx, k_off]
                        acc = acc + v.to(cutlass.Float32)
                    mRowsum_nkl[_g_token, chunk_idx, 0] = acc

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
        mRowsum_nkl: cute.Tensor,
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
          mA_mkl      : A weights in GMEM, shape (M_out, K, G)
          mSFA_mkl    : A/SFA scales in GMEM, shape (M_out, K / sfa_g, G)
          mB_nkl      : B activations in GMEM, shape (N_tokens, K, 1)
          mSFB_nkl    : B/SFB scales in GMEM, shape (N_tokens, K / sfb_g, 1)
          mRowsum_nkl : precomputed rowsum, shape (N_tokens, K / mma_K, 1)
          mC_mnl      : C output view used by MMA, shape (M_out, N_tokens, 1)

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

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # In 2-CTA mode (V-mode size = 2), only the V=0 CTA is the leader and
        # issues the cooperative tcgen05.mma; the V=1 peer CTA participates via
        # the 2-CTA TMA + cluster mbarriers. In 1-CTA mode V-mode is size 1 so
        # every CTA is leader. mma_tile_coord_v MUST be the per-CTA V coord
        # (NOT hardcoded 0) so partition_C / partition_S in the t2r tiled copy
        # give the peer CTA its own M-half of the cooperative accumulator.
        mma_tile_coord_v = block_in_cluster_coord_vmnk[0]
        is_leader_cta = mma_tile_coord_v == 0

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Initialize load2transform pipeline, which tracks the dependencies between TMA's loading
        # of A and B, and the transformation of A and MMA's consumption
        transform_thread_idx = (
            tidx - 32 * self.transform_warp_id[0]
            if tidx >= 32 * self.transform_warp_id[0]
            else tidx
        )
        # In 2-CTA mode, mbarrier arrive counts double because both CTAs in
        # the cluster pair arrive on each barrier (via multicast / DSMEM
        # remote arrive). cta_v_size = 2 in 2-CTA, 1 otherwise.
        cta_v_size = cute.size(cluster_layout_vmnk, mode=[0])

        # One A TMA per stage feeds the single local M tile owned by this CTA.
        a_load2trans_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.a_load2trans_full_mbar_ptr.data_ptr(),
            num_stages=self.num_load2trans_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mcast_ctas_a * 4,  # 4 TRANSFORM warps x per-CTA-N multicast
            ),
            tx_count=self.num_tma_load_bytes_a,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=transform_thread_idx,
            mcast_mode_mn=(1, 0),
            defer_sync=True,
        )

        # Initialize the scale_load2accu pipeline. It tracks one local SFA TMA
        # load -> epilog scale consume per stage. SFB is not a TMA here; the
        # epilog warps stage it directly into BSBC SMEM at each work-tile
        # prologue.
        _scale_tx = self.num_tma_load_bytes_scale
        scale_load2accu_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.a_scale_load2accu_full_mbar_ptr.data_ptr(),
            num_stages=self.num_scale_load2accu_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mcast_ctas_a * len(self.epilog_warp_id),
            ),
            tx_count=_scale_tx,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            mcast_mode_mn=(1, 0),  # multicast for sfa will only happen on the M-mode
            defer_sync=True,
        )

        # Initialize pipeline for tensor B load to MMA.
        # Prefill rowsum-precompute variant: PipelineTmaUmma is used for both
        # 1-CTA and 2-CTA. The epilog reads rowsum workspace instead of staged
        # B, so no AsyncThread B consumer is needed and the MMA warp is the
        # only B pipeline consumer.
        load2mma_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.b_load2mma_full_mbar_ptr.data_ptr(),
            num_stages=self.num_load2trans_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_mcast_ctas_b
            ),
            tx_count=self.num_tma_load_bytes_b,
            cta_layout_vmnk=cluster_layout_vmnk,
            mcast_mode_mn=(0, 1),
            defer_sync=True,
        )

        # Initialize trans2mma pipeline. The TRANSFORM warps convert the
        # local M tile for this CTA, fence the TMEM store, and commit once.
        # In 2-CTA mode the producer group is cluster-wide.
        trans2mma_pipeline = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.a_trans2mma_full_mbar_ptr.data_ptr(),
            num_stages=self.num_trans2mma_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * 4 * cta_v_size,  # bf16 ref pattern: cluster-wide arrives
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Initialize accumulator pipeline. MMA commits once per k-block after
        # the local tcgen05.mma has been issued; the epilog warps wait once,
        # consume the local acc TMEM region, and release once.
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.epilog_warp_id) * cta_v_size
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
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        if warp_idx == self.schedule_warp_id:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(storage.sfb_cluster_mbar.ptr, 1)

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # --- Allocate SMEM tensors ---
        # SMEM staging tensors. Shapes are described in tile coordinates:
        #   sA: raw int4 A for this M tile.
        #   sSFA: bf16 SFA for this M tile.
        #   sB: fp8 B tile consumed by MMA.
        #   sSFB_bsbc: BSBC SFB scratch consumed by tcgen05.cp (MX policy).
        #   sSFB_post: plain BF16 SFB tile consumed by epilog (BF16 policy).
        #   sRowsum_post: fp32 rowsum tile, (cta_tile_n, K / mma_K).
        #   sC: epilog staging before the TMA/SIMT store to C.
        sC = smem.allocate_tensor(
            element_type=self.c_dtype,
            layout=c_smem_layout.outer,
            byte_alignment=self.smem_buffer_align_bytes,
            swizzle=c_smem_layout.inner,
        )
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout.outer,
            byte_alignment=self.smem_buffer_align_bytes,
            swizzle=a_smem_layout.inner,
        )
        sSFA = smem.allocate_tensor(
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
            # BSBC SMEM scratch for HW SFB. Epilog warps fill one BSBC stage
            # per MMA K tile; the leader MMA warp copies the matching stage to
            # TMEM SFB before issuing tcgen05.mma.
            sSFB_bsbc = smem.allocate_tensor(
                element_type=self.sfb_dtype,
                layout=sfb_smem_layout,
                byte_alignment=self.smem_buffer_align_bytes,
            )
        else:
            # Plain token-major SFB staging for the BF16 post-MMA path. Two
            # guard rows let full odd-start tiles use packed int32 cp.async.
            _sfb_n = self.cta_tile_shape_mnk[1]
            _sfb_stage_n = _sfb_n + 2
            _sfb_chunks = self.k_total // self.sfb_granularity_k
            sSFB_post = smem.allocate_tensor(
                element_type=self.sfb_dtype,
                layout=cute.make_layout(
                    (_sfb_stage_n, _sfb_chunks),
                    stride=(1, _sfb_stage_n),
                ),
                byte_alignment=128,
            )
        # Precomputed rowsum SMEM staging - granularity is mma_K (one cell
        # per token per K-tile). BF16 rowsum is unweighted; E8M0 rowsum folds
        # in SFB weighting.
        _rowsum_n: cutlass.Constexpr[int] = self.cta_tile_shape_mnk[1]
        _rowsum_chunks: cutlass.Constexpr[int] = (
            self.k_total // self.cta_tile_shape_mnk[2]
        )
        sRowsum_post = smem.allocate_tensor(
            element_type=self.rowsum_dtype,
            layout=cute.make_layout(
                (_rowsum_n, _rowsum_chunks),
                stride=(1, _rowsum_n),  # token/N-major
            ),
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

        # Compute multicast masks for A/B/SFA TMAs. In 2-CTA mode (use_2cta_instrs),
        # the cp.async.bulk.tensor.cta_group::2 TMA arrives on BOTH CTAs' transaction
        # mbarriers via the multicast accounting - without these masks, each CTA's
        # mbarrier only sees its own TMA's bytes (b_copy_size) but expects the
        # cluster-aggregated total (2 * b_copy_size for B), causing a permanent
        # consumer_wait stall in the MMA warp.
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        if cutlass.const_expr(
            self.is_a_mcast or self.is_b_mcast or self.use_2cta_instrs
        ):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfa_full_mcast_mask = a_full_mcast_mask
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        # Partition global/shared tensor for TMA load A/B
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )

        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )

        thr_mma_leader_cta = tiled_mma.get_slice(0)
        # (MMA, MMA_M, MMA_K, STAGE)
        tCsSFA = thr_mma_leader_cta.partition_A(sSFA)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tAsSFA, tAgSFA = mixed_input_utils.scale_tma_partition(
            tCsSFA,
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
        tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        # Make transformed A tensor in TMEM (single M-tile per CTA).
        # (MMA, MMA_M, MMA_K, STAGE)
        tmem_ptr_transform = cute.recast_ptr(
            tCtAcc_base.iterator + self.num_acc_tmem_cols,
            dtype=self.tiled_mma.op.a_dtype,
        )
        tCrA = cute.make_tensor(
            tmem_ptr_transform,
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

        if cutlass.const_expr(self.is_mxscale_sfb):
            # Make SFB tmem tensor for HW SFB. Per-MMA-tile TMEM region; the
            # tcgen05.cp from BSBC SMEM scratch writes here, then HW SFB reads
            # via tcgen05.mma.scale. The margin covers SFA TMEM extent slack.
            sfb_tmem_offset = (
                sfa_tmem_offset + self.num_sfa_tmem_cols + self.sfb_tmem_margin_cols
            )
            _sfb_tmem_layout_source = self.sfb_smem_layout_per_stage
        else:
            # BF16 SFB is applied in the epilog. The hardware SFB channel is
            # kept as identity and never overwritten after prologue.
            sfb_tmem_offset = sfa_tmem_offset + self.num_sfa_tmem_cols
            _sfb_tmem_layout_source = self.sfa_smem_layout_per_stage
        sfb_tmem_ptr = cute.recast_ptr(
            tmem_ptr + sfb_tmem_offset,
            dtype=self.sf_mma_dtype,
        )
        tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_mma_vec_size,
            _sfb_tmem_layout_source,
        )
        tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

        if cutlass.const_expr(self.is_mxscale_sfb):
            # Per-MMA-tile S2T copy of SFB from BSBC SMEM scratch into TMEM
            # SFB. The MMA warp issues cute.gemm with HW SFB applied via
            # tcgen05.mma.kind::mxf8f6f4.
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t,
                tCtSFB_compact_s2t,
            ) = fp8_utils.mainloop_s2t_copy_and_partition(
                sSFB_bsbc,
                tCtSFB,
                self.sf_mma_dtype,
                self.cta_group,
            )

        # Fill unit scale 1.0 for the SFA MMA channel. Real bf16 SFA is
        # applied later in the epilog; real E8M0 SFB is written per K-tile by
        # tcgen05.cp from BSBC SMEM.

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
                # BF16 post-SFB path: fill TMEM SFB with unit scale once.
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
                # cta_coord_m advances by 1 per CTA in the cluster (V=0,V=1,...).
                # tAgA is built from a thr_mma slice over the cluster_M tile, so
                # its loopM dimension is at cluster_M granularity. Divide by the
                # V-axis size to get the cluster-M index. (1-CTA: V=1, // 1 = identity.)
                _cm = work_tile.cta_coord_m // cute.size(tiled_mma.thr_id.shape)
                tAgA_slice = tAgA[(None, _cm, None, work_tile.group_idx)]
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
                a_load2trans_producer_state.reset_count()
                a_peek_load2trans_empty_status = cutlass.Boolean(1)
                if a_load2trans_producer_state.count < k_tile_cnt:
                    a_peek_load2trans_empty_status = (
                        a_load2trans_pipeline.producer_try_acquire(
                            a_load2trans_producer_state
                        )
                    )
                load2mma_producer_state.reset_count()
                # 2-CTA: TMA backpressure flows through MMA. MMA waits on the
                # SFB sidecar before consumer_release; TMA waits on the
                # pipeline empty state as usual.
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Single A TMA per stage (one M-tile per CTA).
                    a_load2trans_pipeline.producer_acquire(
                        a_load2trans_producer_state, a_peek_load2trans_empty_status
                    )
                    load2mma_pipeline.producer_acquire(load2mma_producer_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, a_load2trans_producer_state.count)],
                        tAsA[(None, a_load2trans_producer_state.index)],
                        tma_bar_ptr=a_load2trans_pipeline.producer_get_barrier(
                            a_load2trans_producer_state
                        ),
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, load2mma_producer_state.count)],
                        tBsB[(None, load2mma_producer_state.index)],
                        tma_bar_ptr=load2mma_pipeline.producer_get_barrier(
                            load2mma_producer_state
                        ),
                        mcast_mask=b_full_mcast_mask,
                    )
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
            # Wait A/B buffer empty
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

            # Scale producer state for the local SFA tile.
            scale_load2accu_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_scale_load2accu_stage
            )
            scale_k_tile_cnt = cute.size(mSFA_mkl.layout.shape[1][1])
            while work_tile.is_valid_tile:
                # ((atom_v, rest_v), RestK)
                # tAgSFA loopM granularity is cluster_M. See note above tAgA.
                _cm = work_tile.cta_coord_m // cute.size(tiled_mma.thr_id.shape)
                tAgSFA_slice = tAgSFA[
                    (
                        None,
                        _cm,
                        None,
                        work_tile.group_idx,
                    )
                ]

                # Filter zeros in rest mode
                rest_filtered = cute.filter_zeros(tAgSFA_slice[(0, None)].layout)
                tAgSFA_slice_filtered = cute.make_tensor(
                    tAgSFA_slice.iterator,
                    cute.make_layout(
                        (tAgSFA_slice.layout[0].shape, rest_filtered.shape),
                        stride=(
                            tAgSFA_slice.layout[0].stride,
                            rest_filtered.stride,
                        ),
                    ),
                )

                # No SFB load here; regular-layout SFB is staged by the epilog
                # warps and remapped to BSBC by the MMA warp.

                scale_load2accu_producer_state.reset_count()
                peek_scale_load2accu_empty_status = cutlass.Boolean(1)
                if scale_load2accu_producer_state.count < scale_k_tile_cnt:
                    peek_scale_load2accu_empty_status = (
                        scale_load2accu_pipeline.producer_try_acquire(
                            scale_load2accu_producer_state
                        )
                    )
                for k_tile in cutlass.range(0, scale_k_tile_cnt, 1, unroll=1):
                    scale_load2accu_pipeline.producer_acquire(
                        scale_load2accu_producer_state,
                        peek_scale_load2accu_empty_status,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice_filtered[
                            (None, scale_load2accu_producer_state.count)
                        ],
                        tAsSFA[(None, scale_load2accu_producer_state.index)],
                        tma_bar_ptr=scale_load2accu_pipeline.producer_get_barrier(
                            scale_load2accu_producer_state
                        ),
                        mcast_mask=sfa_full_mcast_mask,
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

            # Wait until epilog consumers have released all scale buffers.
            scale_load2accu_pipeline.producer_tail(scale_load2accu_producer_state)

        # Specialized TRANSFORM warps - single-tile per CTA.
        # Per k_tile the body does:
        #   - consumer_wait on a_load2trans_pipeline (SMEM landed)
        #   - producer_acquire on trans2mma_pipeline (covers TMEM A)
        #   - Transform -> write TMEM A (in-place store)
        #   - fence_view_async_tmem_store
        #   - trans2mma producer_commit signals MMA
        #   - a_load2trans consumer_release frees the SMEM stage
        if self.transform_warp_id[0] <= warp_idx <= self.transform_warp_id[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_transform_warps)
            transform_local_tidx = (
                tidx - 32 * self.transform_warp_id[0]
            )  # [0, 128) per group

            # Per-tile partitions - destination is always TMEM.
            src_copy_a, dst_copy_a, tAsA_input, tAsA_transform = (
                mixed_input_utils.transform_partition(
                    tcgen05.OperandSource.TMEM,
                    TransformMode.ConvertScale,
                    copy_atom_a_input,
                    copy_atom_a_transform,
                    sA,
                    tCrA,
                    transform_local_tidx,
                )
            )

            tArA_load = cute.make_rmem_tensor(
                cute.append(tAsA_input[(None, None, None, None, 0)].shape, 1),
                tAsA_input.element_type,
            )
            tArA_transform = cute.make_rmem_tensor(
                cute.append(tAsA_input[(None, None, None, None, 0)].shape, 1),
                self.a_transformed_dtype,
            )
            transform_tiler_size = min(
                cute.size(cute.coalesce(tAsA_input.layout), mode=[0]), 32
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

            # Single consumer state for this CTA's local A tile.
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
                    a_load2trans_pipeline.consumer_wait(a_load2trans_consumer_state)
                    trans2mma_pipeline.producer_acquire(trans2mma_producer_state)

                    tAsA_input_slice = fp8_utils.divide_tensor_by_tiler(
                        tAsA_input[
                            (None, None, None, None, a_load2trans_consumer_state.index)
                        ],
                        transform_tiler,
                    )
                    tArA_load_slice = fp8_utils.divide_tensor_by_tiler(
                        tArA_load[(None, None, None, None, 0)],
                        transform_tiler,
                    )
                    tArA_transform_buffer = tArA_transform[(None, None, None, None, 0)]
                    tArA_transform_slice = fp8_utils.divide_tensor_by_tiler(
                        tArA_transform_buffer, transform_tiler
                    )

                    for idx in cutlass.range_constexpr(
                        cute.size(tArA_load_slice, mode=[1])
                    ):
                        cute.autovec_copy(
                            tAsA_input_slice[(None, idx)],
                            tArA_load_slice[(None, idx)],
                        )

                    for idx in cutlass.range_constexpr(
                        cute.size(tArA_load_slice, mode=[1])
                    ):
                        # Biased int4->fp8 conversion (1 lop3.b32 LUT 0x28).
                        # Produces a biased accumulator; epilog applies the
                        # dtype-selected scale and rowsum correction.
                        tensor_transformed = mixed_input_utils.cvt_tensor_a_biased(
                            tArA_load_slice[(None, idx)],
                            self.a_transformed_dtype,
                        )
                        tArA_transform_slice[(None, idx)].store(tensor_transformed)

                    mixed_input_utils.store_transformed_a(
                        tArA_transform_buffer,
                        tAsA_transform[
                            (None, None, None, None, trans2mma_producer_state.index)
                        ],
                        dst_copy_a,
                    )

                    cute.arch.fence_view_async_tmem_store()
                    trans2mma_pipeline.producer_commit(trans2mma_producer_state)
                    trans2mma_producer_state.advance()

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
            # trans2mma consumer state - single barrier signals TMEM A is
            # ready (transform warps commit once per k_tile).
            trans2mma_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_trans2mma_stage
            )
            load2mma_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_load2trans_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            sfb_cluster_phase = cutlass.Int32(0)
            while work_tile.is_valid_tile:
                load2mma_consumer_state.reset_count()
                trans2mma_consumer_state.reset_count()
                if cutlass.const_expr(self.is_mxscale_sfb):
                    # Epilog warps publish regular-GMEM SFB directly into per-K
                    # BSBC SFB stages. The MMA warp participates in the local
                    # barrier before copying the stages to TMEM SFB.
                    self.sfb_smem_ready_barrier.arrive_and_wait()
                    if cutlass.const_expr(self.use_2cta_instrs):
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(
                                storage.sfb_cluster_mbar.ptr,
                                cta_rank_in_cluster ^ 1,
                            )
                        cute.arch.mbarrier_wait(
                            storage.sfb_cluster_mbar.ptr, sfb_cluster_phase
                        )
                        sfb_cluster_phase = sfb_cluster_phase ^ 1
                peek_trans2mma_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    if trans2mma_consumer_state.count < k_tile_cnt:
                        peek_trans2mma_full_status = (
                            trans2mma_pipeline.consumer_try_wait(
                                trans2mma_consumer_state
                            )
                        )
                if is_leader_cta:
                    # Per-tile-scale: each MMA tile uses accumulate=False and
                    # commits the acc pipeline ONCE per MMA tile.
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        acc_pipeline.producer_acquire(acc_producer_state)
                        # (MMA, MMA_M, MMA_N)
                        tCtAcc = tCtAcc_base[
                            (None, None, None, acc_producer_state.index)
                        ]

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
                            # tcgen05.cp.cta_group::N from BSBC SMEM -> TMEM SFB.
                            s2t_stage_coord = (
                                None,
                                None,
                                None,
                                None,
                                k_tile,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfb,
                                tCsSFB_compact_s2t[s2t_stage_coord],
                                tCtSFB_compact_s2t,
                            )

                            # Block-scaled MMA per k_tile (kind::mxf8f6f4) using
                            # variadic operand-list API.
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                [tCrA[kblock_coord_a], tCtSFA],
                                [tCrB[kblock_coord_b], tCtSFB],
                                tCtAcc,
                            )
                        else:
                            # Regular fp8 MMA; SFB is applied post-MMA in the
                            # epilog and TMEM scale channels are identity.
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord_a],
                                tCrB[kblock_coord_b],
                                tCtAcc,
                            )

                        trans2mma_pipeline.consumer_release(trans2mma_consumer_state)
                        trans2mma_consumer_state.advance()

                        # PipelineTmaUmma - single MMA consumer in both 1-CTA
                        # and 2-CTA prefill paths.
                        load2mma_pipeline.consumer_release(load2mma_consumer_state)
                        load2mma_consumer_state.advance()

                        peek_trans2mma_full_status = cutlass.Boolean(1)
                        if trans2mma_consumer_state.count < k_tile_cnt:
                            peek_trans2mma_full_status = (
                                trans2mma_pipeline.consumer_try_wait(
                                    trans2mma_consumer_state
                                )
                            )

                        # Per-tile commit - fresh acc ready.
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
            scale_view_as_C = cute.make_tensor(
                sSFA.iterator,
                scale_view_as_C_layout,
            )
            # Partition for epilogue and accumulator update
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
                tTR_rAcc_final,
                tTR_sScale,
            ) = fp8_utils.epilog_and_acc_update_tmem_copy_and_partition(
                epi_tidx,
                tCtAcc_base,
                tCgC,
                scale_view_as_C,
                epi_tile,
                self.cta_tile_shape_mnk,
                self.c_layout,
                self.c_dtype,
                self.acc_dtype,
                self.use_2cta_instrs,
            )

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)

            # The rowsum precompute consumes B before the main GEMM, so the
            # epilog does not drain staged B or act as a B pipeline consumer.

            tiled_copy_r2s, tRS_rC, tRS_sC = (
                mixed_input_utils.epilog_smem_copy_and_partition(
                    self.c_layout,
                    self.c_dtype,
                    self.acc_dtype,
                    tiled_copy_t2r,
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
                    tiled_copy_t2r,
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
            thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
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
            # cp.async copy atom for per-thread rowsum GMEM->SMEM load.
            _rowsum_cpasync_atom = cute.make_copy_atom(
                cpasync.CopyG2SOp(),
                cutlass.Int32,
                num_bits_per_copy=32,
            )
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
                tTR_rAcc_final.fill(0.0)

                tTR_rScale = cute.make_rmem_tensor(
                    cute.slice_(tTR_sScale, (None, None, None, 0, None, 0)).shape,
                    self.sfa_dtype,
                )
                _num_epilog_thr: cutlass.Constexpr[int] = 32 * len(self.epilog_warp_id)
                if cutlass.const_expr(self.is_mxscale_sfb):
                    self._stage_sfb_gmem_to_bsbc(
                        mSFB_nkl,
                        sSFB_bsbc,
                        epi_tidx,
                        work_tile.coord_n,
                        work_tile.distance_to_boundary,
                        _num_epilog_thr,
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.sfb_smem_ready_barrier.arrive_and_wait()
                else:
                    _sfb_num_chunks: cutlass.Constexpr[int] = (
                        self.k_total // self.sfb_granularity_k
                    )
                    _sfb_n: cutlass.Constexpr[int] = self.cta_tile_shape_mnk[1]
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
                        True,
                    )

                if cutlass.const_expr(self.use_pdl_rowsum):
                    cute.arch.griddepcontrol_wait()

                # Cooperative cp.async load of rowsum into sRowsum_post.
                # Layout (cta_tile_n, num_rowsum_chunks) fp32 token/N-major; each
                # int32 = one fp32 rowsum value. One cp.async per (n_local,
                # k_tile_idx) cell. fp32 is needed for the biased correction
                # term.
                _rowsum_total: cutlass.Constexpr[int] = _rowsum_n * _rowsum_chunks
                _rowsum_per_thr_loads: cutlass.Constexpr[int] = (
                    _rowsum_total + _num_epilog_thr - 1
                ) // _num_epilog_thr
                _mRowsum_nkl_i32 = cute.recast_tensor(mRowsum_nkl, cutlass.Int32)
                _sRowsum_post_i32 = cute.recast_tensor(sRowsum_post, cutlass.Int32)
                for _li in cutlass.range_constexpr(_rowsum_per_thr_loads):
                    _idx = epi_tidx + _li * _num_epilog_thr
                    if _idx < _rowsum_total:
                        _rs_chunk_idx = _idx // _rowsum_n
                        _rs_n_local = _idx % _rowsum_n
                        _rs_n_global = work_tile.coord_n + _rs_n_local
                        _rs_dst_view = fp8_utils.make_single_element_tensor_view(
                            _sRowsum_post_i32, (_rs_n_local, _rs_chunk_idx)
                        )
                        if (_rs_n_local < work_tile.distance_to_boundary) and (
                            _rs_n_global < mRowsum_nkl.shape[0]
                        ):
                            _rs_src_view = fp8_utils.make_single_element_tensor_view(
                                _mRowsum_nkl_i32, (_rs_n_global, _rs_chunk_idx, 0)
                            )
                            cute.copy(_rowsum_cpasync_atom, _rs_src_view, _rs_dst_view)
                        else:
                            _sRowsum_post_i32[(_rs_n_local, _rs_chunk_idx)] = (
                                cutlass.Int32(0)
                            )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                # Publish rowsum loads across the four epilog warps. HW SFB is
                # already staged through the SFB-specific barrier above, so the
                # MMA warp does not participate in this rowsum barrier.
                self.epilog_sync_barrier.arrive_and_wait()

                # -------------------------------------------------------
                # Per-MMA-tile rowsum correction + acc-rescale loop.
                #
                # For each MMA tile (= cta_tile_k K-elements):
                #   1. Wait for the per-MMA-tile s32 acc TMEM (acc pipeline
                #      now fires per MMA tile - accumulate=False MMA).
                #   2. On SFA chunk boundary, autovec_copy the SFA SMEM ->
                #      RMEM register fragment (kept across MMA tiles within
                #      a chunk).
                #   3. Accumulate the biased correction. BF16 multiplies the
                #      unweighted rowsum correction by SFB in the epilog; E8M0
                #      uses the preweighted rowsum matching HW-SFB-scaled MMA.
                #
                # Total acc-pipeline events: k_tile_cnt (NOT scale_k_tile_cnt).
                # -------------------------------------------------------
                scale_consumer_state.reset_count()
                acc_consumer_state.reset_count()
                if cutlass.const_expr(self.is_mxscale_sfb):
                    if cutlass.const_expr(num_k_tiles_per_sfa == 1):
                        # Fast path: every MMA K-tile has its own SFA tile, so
                        # fuse T2R, biased rowsum correction, and SFA application.
                        for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                            scale_load2accu_pipeline.consumer_wait(scale_consumer_state)
                            _scale_stage_idx = scale_consumer_state.index
                            tTR_sScale_slice = cute.slice_(
                                tTR_sScale,
                                (None, None, None, 0, None, _scale_stage_idx),
                            )
                            cute.autovec_copy(tTR_sScale_slice, tTR_rScale)

                            acc_pipeline.consumer_wait(acc_consumer_state)
                            _acc_stage_idx = acc_consumer_state.index

                            self._correct_tile_precomputed_rowsum(
                                tTR_tAcc_base,
                                tiled_copy_t2r,
                                tTR_rAcc,
                                tTR_rAcc_final,
                                tTR_rScale,
                                sRowsum_post,
                                k_tile,
                                m_thr_offset,
                                _acc_stage_idx,
                            )

                            with cute.arch.elect_one():
                                acc_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()
                            scale_load2accu_pipeline.consumer_release(
                                scale_consumer_state
                            )
                            scale_consumer_state.advance()
                    else:
                        # Chunked path: several MMA K-tiles share one SFA tile,
                        # so defer SFA application until the chunk boundary.
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
                        chunk_acc_sum = cute.make_rmem_tensor(
                            tTR_rAcc_final.shape, self.acc_dtype
                        )
                        chunk_acc_sum.fill(0.0)

                        for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                            # Per-MMA-tile acc consume + biased-correction accumulate.
                            acc_pipeline.consumer_wait(
                                acc_consumer_state, peek_acc_full_status
                            )
                            _acc_stage_idx = acc_consumer_state.index

                            self._chunked_acc_accumulate(
                                tTR_tAcc_base,
                                tiled_copy_t2r,
                                tTR_rAcc,
                                chunk_acc_sum,
                                sRowsum_post,
                                k_tile,
                                m_thr_offset,
                                _acc_stage_idx,
                            )

                            with cute.arch.elect_one():
                                acc_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()
                            peek_acc_full_status = cutlass.Boolean(1)
                            if acc_consumer_state.count < k_tile_cnt:
                                peek_acc_full_status = acc_pipeline.consumer_try_wait(
                                    acc_consumer_state
                                )

                            _next_k_tile = k_tile + 1
                            _at_chunk_close = (_next_k_tile % num_k_tiles_per_sfa) == 0
                            if _at_chunk_close:
                                scale_load2accu_pipeline.consumer_wait(
                                    scale_consumer_state, peek_scale_full_status
                                )
                                _scale_stage_idx = scale_consumer_state.index
                                tTR_sScale_slice = cute.slice_(
                                    tTR_sScale,
                                    (None, None, None, 0, None, _scale_stage_idx),
                                )
                                cute.autovec_copy(tTR_sScale_slice, tTR_rScale)
                                self._chunked_close_apply(
                                    chunk_acc_sum,
                                    tTR_rAcc_final,
                                    tTR_rScale,
                                    m_thr_offset,
                                )
                                scale_load2accu_pipeline.consumer_release(
                                    scale_consumer_state
                                )
                                scale_consumer_state.advance()
                                peek_scale_full_status = cutlass.Boolean(1)
                                if scale_consumer_state.count < scale_k_tile_cnt:
                                    peek_scale_full_status = (
                                        scale_load2accu_pipeline.consumer_try_wait(
                                            scale_consumer_state
                                        )
                                    )
                else:
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
                    chunk_acc_sum = cute.make_rmem_tensor(
                        tTR_rAcc_final.shape, self.acc_dtype
                    )
                    chunk_acc_sum.fill(0.0)
                    num_k_tiles_per_sfb_local = (
                        self.sfb_granularity_k // self.cta_tile_shape_mnk[2]
                    )

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        _k_chunk_idx_sfb = k_tile // num_k_tiles_per_sfb_local
                        if cutlass.const_expr(num_k_tiles_per_sfa == 1):
                            acc_pipeline.consumer_wait(acc_consumer_state)
                        else:
                            acc_pipeline.consumer_wait(
                                acc_consumer_state, peek_acc_full_status
                            )
                        _acc_stage_idx = acc_consumer_state.index

                        self._chunked_acc_accumulate_bf16_post_sfb(
                            tTR_tAcc_base,
                            tiled_copy_t2r,
                            tTR_rAcc,
                            chunk_acc_sum,
                            sSFB_post,
                            sRowsum_post,
                            _sfb_smem_row_offset,
                            _k_chunk_idx_sfb,
                            k_tile,
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

                        # On SFA chunk close: load SFA, apply running update,
                        # reset chunk accumulator, release SFA.
                        _next_k_tile = k_tile + 1
                        _at_chunk_close = (_next_k_tile % num_k_tiles_per_sfa) == 0
                        if _at_chunk_close:
                            if cutlass.const_expr(num_k_tiles_per_sfa == 1):
                                scale_load2accu_pipeline.consumer_wait(
                                    scale_consumer_state
                                )
                            else:
                                scale_load2accu_pipeline.consumer_wait(
                                    scale_consumer_state, peek_scale_full_status
                                )
                            _scale_stage_idx = scale_consumer_state.index
                            tTR_sScale_slice = cute.slice_(
                                tTR_sScale,
                                (None, None, None, 0, None, _scale_stage_idx),
                            )
                            cute.autovec_copy(tTR_sScale_slice, tTR_rScale)
                            self._chunked_close_apply(
                                chunk_acc_sum,
                                tTR_rAcc_final,
                                tTR_rScale,
                                m_thr_offset,
                            )
                            scale_load2accu_pipeline.consumer_release(
                                scale_consumer_state
                            )
                            scale_consumer_state.advance()
                            if cutlass.const_expr(num_k_tiles_per_sfa != 1):
                                peek_scale_full_status = cutlass.Boolean(1)
                                if scale_consumer_state.count < scale_k_tile_cnt:
                                    peek_scale_full_status = (
                                        scale_load2accu_pipeline.consumer_try_wait(
                                            scale_consumer_state
                                        )
                                    )
                num_prev_subtiles = fp8_utils.store_prefill_accumulator_tile(
                    tTR_rAcc_final,
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
                    self.use_simt_store,
                    self.c_layout.is_n_major_c(),
                    cute.size(tiled_mma.thr_id.shape),
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
    def _chunked_acc_accumulate_bf16_post_sfb(
        self,
        tTR_tAcc_base: cute.Tensor,
        tiled_copy_t2r,
        tTR_rAcc: cute.Tensor,
        chunk_acc_sum: cute.Tensor,
        sSFB_post: cute.Tensor,
        sRowsum_post: cute.Tensor,
        sfb_smem_row_offset: cutlass.Int32,
        k_chunk_idx_sfb: cutlass.Int32,
        k_tile_idx: cutlass.Int32,
        m_thr_offset: cute.Tensor,
        acc_stage_idx: cutlass.Int32,
    ):
        """Per-MMA-tile T2R + biased correction for BF16 post-MMA SFB."""
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
                n_local_i = m_thr_sub_k[(i)][1]
                sfb_i = sSFB_post[
                    (n_local_i + sfb_smem_row_offset, k_chunk_idx_sfb)
                ].to(self.acc_dtype)
                rowsum_i = sRowsum_post[(n_local_i, k_tile_idx)].to(self.acc_dtype)
                corrected = (
                    cutlass.Float32(512.0) * acc_vec[(i)]
                    - cutlass.Float32(8.0) * rowsum_i
                )
                chunk_acc_subtile[i] = chunk_acc_subtile[(i)] + sfb_i * corrected

    @cute.jit
    def _correct_tile_precomputed_rowsum(
        self,
        tTR_tAcc_base: cute.Tensor,
        tiled_copy_t2r,
        tTR_rAcc: cute.Tensor,
        tTR_rAcc_final: cute.Tensor,
        tTR_rScale: cute.Tensor,
        sRowsum_post: cute.Tensor,
        k_tile_idx: cutlass.Int32,
        m_thr_offset: cute.Tensor,
        acc_stage_idx: cutlass.Int32,
    ):
        """Single-SFA-tile fast path:
            running += sfa * (512 * acc - 8 * weighted_rowsum).

        Used when num_k_tiles_per_sfa == 1, so each MMA K-tile has exactly one
        matching SFA tile already loaded into RMEM by the caller.
        """
        tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_idx)]
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])

        for subtile_idx in cutlass.range_constexpr(subtile_cnt):
            tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
            running_subtile = tTR_rAcc_final[(None, None, None, subtile_idx)]
            scale_subtile = tTR_rScale[(None, None, None, subtile_idx)]
            acc_vec = tTR_rAcc.load()
            scale = scale_subtile.load().to(self.acc_dtype)
            m_thr_sub_k = m_thr_offset[(None, None, None, subtile_idx)]
            for i in cutlass.range(cute.size(running_subtile.shape), unroll_full=True):
                n_local_i = m_thr_sub_k[(i)][1]
                rowsum_w_i = sRowsum_post[(n_local_i, k_tile_idx)].to(self.acc_dtype)
                scale_i = scale[(i)]
                corrected_i = scale_i * (
                    cutlass.Float32(512.0) * acc_vec[(i)]
                    - cutlass.Float32(8.0) * rowsum_w_i
                )
                running_subtile[i] = running_subtile[(i)] + corrected_i

    @cute.jit
    def _chunked_acc_accumulate(
        self,
        tTR_tAcc_base: cute.Tensor,
        tiled_copy_t2r,
        tTR_rAcc: cute.Tensor,
        chunk_acc_sum: cute.Tensor,
        sRowsum_post: cute.Tensor,
        k_tile_idx: cutlass.Int32,
        m_thr_offset: cute.Tensor,
        acc_stage_idx: cutlass.Int32,
    ):
        """Per-MMA-tile T2R + biased correction with sfb-weighted rowsum.

        The HW MMA (`tcgen05.mma.kind::mxf8f6f4`) already applies sfb
        per-sub-block at sf_vec_size=32 grain inside the MMA. The matching
        sfb-weighted rowsum
            R_w[t, k_tile] = sum_{sub in K_tile} sfb[t, sub] * sum_{k in sub} B[t, k]
        is precomputed by the rowsum kernel (see `_rowsum_kernel`). Per K-tile
        correction collapses to one read + one fmsub per element:
            corrected = 512*acc_vec - 8*R_w
            chunk_acc_sum[i] += corrected
        Independent of sfb_g - fast path even at mxfp8-native sfb=32.
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
                n_local_i = m_thr_sub_k[(i)][1]
                rowsum_w_i = sRowsum_post[(n_local_i, k_tile_idx)].to(self.acc_dtype)
                chunk_acc_subtile[i] = chunk_acc_subtile[(i)] + (
                    cutlass.Float32(512.0) * acc_vec[(i)]
                    - cutlass.Float32(8.0) * rowsum_w_i
                )

    @cute.jit
    def _chunked_close_apply(
        self,
        chunk_acc_sum: cute.Tensor,
        tTR_rAcc_final: cute.Tensor,
        tTR_rScale: cute.Tensor,
        m_thr_offset: cute.Tensor,
    ):
        """Per-SFA-chunk: running += sfa*chunk_acc_sum.

        The per-MMA-tile loop has already folded in the biased-fp8 correction
        (`512*acc - 8*weighted_rowsum`). HW SFB is already applied inside the
        MMA. This close step applies SFA once per element at the SFA-chunk
        boundary and resets chunk_acc_sum.
        """
        subtile_cnt = cute.size(tTR_rAcc_final.shape, mode=[3])

        for subtile_idx in cutlass.range_constexpr(subtile_cnt):
            chunk_acc_subtile = chunk_acc_sum[(None, None, None, subtile_idx)]
            running_subtile = tTR_rAcc_final[(None, None, None, subtile_idx)]
            scale_subtile = tTR_rScale[(None, None, None, subtile_idx)]
            scale = scale_subtile.load().to(self.acc_dtype)
            for i in cutlass.range(
                cute.size(chunk_acc_subtile.shape), unroll_full=True
            ):
                scale_i = scale[(i)]
                running_subtile[i] = (
                    running_subtile[(i)] + scale_i * chunk_acc_subtile[(i)]
                )
                chunk_acc_subtile[i] = cutlass.Float32(0.0)

    @cute.jit
    def _stage_sfb_gmem_to_bsbc(
        self,
        mSFB_nkl: cute.Tensor,
        sSFB_bsbc: cute.Tensor,
        tidx: cutlass.Int32,
        n_origin: cutlass.Int32,
        distance_to_boundary: cutlass.Int32,
        num_threads: cutlass.Constexpr[int],
    ):
        """Stage regular E8M0 SFB GMEM directly into BSBC SMEM stages.

        The four epilog warps cooperatively fill all K-tile stages for the
        current token tile. SFB is token-owned, not expert-owned, so even
        ragged expert tails initialize the full MMA-N tile. The MMA consumes
        all 32 token lanes before the epilogue masks stores, and leaving
        out-of-expert SFB lanes undefined can corrupt the peer CTA half in
        2-CTA HW-SFB mode.
        """
        cta_tile_n: cutlass.Constexpr[int] = self.cta_tile_shape_mnk[1]
        cta_tile_k: cutlass.Constexpr[int] = self.cta_tile_shape_mnk[2]
        num_sfb_slots_per_chunk: cutlass.Constexpr[int] = (
            self.sfb_granularity_k // self.sf_vec_size
        )
        num_sfb_slots_per_mma_tile: cutlass.Constexpr[int] = (
            cta_tile_k // self.sf_vec_size
        )
        num_sfb_slots_total: cutlass.Constexpr[int] = self.k_total // self.sf_vec_size
        num_k_tiles: cutlass.Constexpr[int] = (
            num_sfb_slots_total // num_sfb_slots_per_mma_tile
        )
        num_sfb_chunks_total: cutlass.Constexpr[int] = (
            self.k_total // self.sfb_granularity_k
        )
        total_cells: cutlass.Constexpr[int] = cta_tile_n * num_sfb_slots_total
        num_passes: cutlass.Constexpr[int] = (
            total_cells + num_threads - 1
        ) // num_threads

        if (distance_to_boundary >= cta_tile_n) and (
            n_origin + cta_tile_n <= mSFB_nkl.shape[0]
        ):
            for _pass in cutlass.range_constexpr(num_passes):
                cell_idx = tidx + _pass * num_threads
                if cell_idx < total_cells:
                    n_local, k_slot_global = cute.idx2crd(
                        cell_idx, (cta_tile_n, num_sfb_slots_total)
                    )
                    k_slot, stage_idx = cute.idx2crd(
                        k_slot_global, (num_sfb_slots_per_mma_tile, num_k_tiles)
                    )
                    _, k_chunk_global = cute.idx2crd(
                        k_slot_global,
                        (num_sfb_slots_per_chunk, num_sfb_chunks_total),
                    )
                    n_inner, n_outer = cute.idx2crd(
                        n_local, (32, cute.ceil_div(cta_tile_n, 32))
                    )
                    val = mSFB_nkl[(n_origin + n_local, k_chunk_global, 0)]
                    if cutlass.const_expr(num_sfb_slots_per_mma_tile <= 4):
                        # For mma_K=128 there are exactly four HW scale slots
                        # per MMA tile, and make_smem_layout_sfb exposes the
                        # MMA-K coordinate as a scalar 0..3.
                        sSFB_bsbc[
                            (
                                ((n_inner, n_outer), (0, 0)),
                                0,
                                k_slot,
                                stage_idx,
                            )
                        ] = val
                    else:
                        # For mma_K=256 the MMA-K coordinate is hierarchical.
                        k_inner, k_outer = cute.idx2crd(
                            k_slot, (4, num_sfb_slots_per_mma_tile // 4)
                        )
                        sSFB_bsbc[
                            (
                                ((n_inner, n_outer), (0, 0)),
                                0,
                                (k_inner, k_outer),
                                stage_idx,
                            )
                        ] = val
        else:
            for _pass in cutlass.range_constexpr(num_passes):
                cell_idx = tidx + _pass * num_threads
                if cell_idx < total_cells:
                    n_local, k_slot_global = cute.idx2crd(
                        cell_idx, (cta_tile_n, num_sfb_slots_total)
                    )
                    k_slot, stage_idx = cute.idx2crd(
                        k_slot_global, (num_sfb_slots_per_mma_tile, num_k_tiles)
                    )
                    _, k_chunk_global = cute.idx2crd(
                        k_slot_global,
                        (num_sfb_slots_per_chunk, num_sfb_chunks_total),
                    )
                    n_inner, n_outer = cute.idx2crd(
                        n_local, (32, cute.ceil_div(cta_tile_n, 32))
                    )
                    n_global = n_origin + n_local
                    if n_global < mSFB_nkl.shape[0]:
                        val = mSFB_nkl[(n_global, k_chunk_global, 0)]
                        if cutlass.const_expr(num_sfb_slots_per_mma_tile <= 4):
                            sSFB_bsbc[
                                (
                                    ((n_inner, n_outer), (0, 0)),
                                    0,
                                    k_slot,
                                    stage_idx,
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
                                    stage_idx,
                                )
                            ] = val
                    else:
                        val = mSFB_nkl[(n_origin, k_chunk_global, 0)]
                        if cutlass.const_expr(num_sfb_slots_per_mma_tile <= 4):
                            sSFB_bsbc[
                                (
                                    ((n_inner, n_outer), (0, 0)),
                                    0,
                                    k_slot,
                                    stage_idx,
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
                                    stage_idx,
                                )
                            ] = val

    @staticmethod
    def _compute_stages_and_tmem_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: tuple[int, int, int],
        cta_tile_shape_mnk: tuple[int, int, int],
        k_total: int,
        epi_tile: cute.Tile,
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        sfa_dtype: type[cutlass.Numeric],
        sfb_dtype: type[cutlass.Numeric],
        rowsum_dtype: type[cutlass.Numeric],
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        scale_granularity_m: int,
        sfa_granularity_k: int,
        sfb_granularity_k: int,
        sf_vec_size: int,
        is_mxscale_sfb: bool,
        smem_extra_bytes: int = 0,
    ) -> tuple[int, int, int, int, int, int, int, int, int, int]:
        """
        Compute pipeline stages and TMEM column allocation configurations.

        A is assumed TMEM-sourced (K-major A invariant). SMEM-source paths
        were removed.
        """
        # --- TMEM column budgets per stage ---
        # Accumulator D for the single local CTA tile, aligned to 2 columns.
        acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
        tCtAcc_stage1 = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))
        num_tmem_acc_col_per_stage = cute.round_up(
            tcgen05.find_tmem_tensor_col_offset(tCtAcc_stage1), 2
        )

        # Scale factors: SFA and SFB share the same per-stage TMEM footprint.
        # MX HW-SFB rejects sub-BSBC MMA-K tiles; the shared helper also keeps
        # internal callers from allocating a zero-column scale region.
        num_tmem_sf_col_per_stage = fp8_utils.blockscaled_scale_tmem_cols(
            cta_tile_shape_mnk[2], sf_vec_size
        )
        num_tmem_sfa_col_per_stage = num_tmem_sf_col_per_stage
        num_tmem_cols_sfb_per_stage = num_tmem_sf_col_per_stage

        # Converted A (TMEM-sourced): one 32-bit TMEM column holds 32 /
        # a_dtype.width elements. Single M-tile per CTA, aligned to 4.
        num_a_elts_per_tmem_col = 32 // tiled_mma.op.a_dtype.width
        num_tmem_cols_a_per_stage = cute.round_up(
            cta_tile_shape_mnk[2] // num_a_elts_per_tmem_col,
            4,
        )

        # SFA is filled once with unit 1.0 at prologue. SFB reuses one TMEM
        # scale region; each K tile overwrites it from the prebuilt BSBC stage
        # immediately before the matching MMA.
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
        if num_tmem_acc_col_per_stage <= 32:
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

        # SFB and rowsum sidecar buffers are fixed per work tile, not per A/B
        # pipeline stage. Reserve them up front so large-K shapes auto-reduce
        # pipeline stages instead of over-allocating dynamic SMEM at launch.
        k_tile_count = max(1, k_total // cta_tile_shape_mnk[2])
        if is_mxscale_sfb:
            sfb_smem_layout_all_stages = blockscaled_utils.make_smem_layout_sfb(
                tiled_mma, mma_tiler_mnk, sf_vec_size, k_tile_count
            )
            sfb_sidecar_bytes = fp8_utils.aligned_smem_bytes(
                sfb_dtype,
                sfb_smem_layout_all_stages,
            )
        else:
            sfb_chunks = max(1, k_total // sfb_granularity_k)
            sfb_stage_n = cta_tile_shape_mnk[1] + 2
            sfb_sidecar_bytes = fp8_utils.aligned_smem_bytes(
                sfb_dtype,
                cute.make_layout(
                    (sfb_stage_n, sfb_chunks),
                    stride=(1, sfb_stage_n),
                ),
                128,
            )
        rowsum_bytes = fp8_utils.aligned_smem_bytes(
            rowsum_dtype,
            cute.make_layout(
                (cta_tile_shape_mnk[1], k_tile_count),
                stride=(1, cta_tile_shape_mnk[1]),
            ),
            128,
        )

        carveout_smem_bytes = (
            bytes_per_pipeline_stage * accumulator_stage_count
            + a_scale_bytes
            + c_bytes
            + tile_info_bytes
            + sfb_sidecar_bytes
            + rowsum_bytes
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
        # Single A per stage - single-tile per CTA.
        # load2trans stages auto-reduce to fit the ~228 KB SMEM budget.
        a_load_bytes_per_stage = fp8_utils.aligned_smem_bytes(
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
        # Combined A+B load bytes per stage (including mbarrier overhead).
        # SFB BSBC scratch is fixed per K-tile and is accounted in the carveout
        # above, not multiplied by pipeline depth.
        ab_load_bytes_per_stage = int(
            a_load_bytes_per_stage
            + b_load_bytes_per_stage
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
        max_transform2mma_stage_count = min(
            transform2mma_stage_count_tmem_limited,
            transform2mma_stage_count_smem_limited,
        )
        transform2mma_stage_count = max_transform2mma_stage_count

        # load2transform stage count: remaining SMEM after trans2mma stages.
        max_load2transform_stage_count = (
            smem_capacity
            - carveout_smem_bytes
            - (transform2mma_stage_count * a_transform_bytes_per_stage)
        ) // ab_load_bytes_per_stage
        load2transform_stage_count = max_load2transform_stage_count

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
        if is_mxscale_sfb:
            # HW-SFB reuses one TMEM scale region; each K tile overwrites it
            # from the prebuilt BSBC stage immediately before MMA.
            num_tmem_sfb_cols = num_tmem_cols_sfb_per_stage
        else:
            # BF16 post-SFB keeps the hardware SFB channel at identity for all
            # trans2mma stages.
            num_tmem_sfb_cols = transform2mma_stage_count * num_tmem_cols_sfb_per_stage

        # Keep C staging small. The generic heuristic burns all leftover SMEM
        # on extra C buffers, which forced this persistent prefill kernel to
        # one CTA per SM (~224 KiB dynamic SMEM). Two C stages are enough for
        # store buffering here and leave occupancy headroom.

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

        use_2cta_instrs = cluster_shape_mn[0] == 2
        if not mixed_input_utils.is_valid_mma_tiler_and_cluster_shape(
            mma_tiler, cluster_shape_mn, use_2cta_instrs
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
    max_active_clusters_override: Optional[int] = None,
    use_pdl_rowsum: bool = False,
    use_simt_store: bool = False,
):
    if sfb_dtype not in [cutlass.BFloat16, cutlass.Float8E8M0FNU]:
        raise ValueError(f"Unsupported prefill SFB dtype: {sfb_dtype}")

    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")

    # Pack (m, n, k, g) into mnkl - consumed as (M_out, N_tokens, K, G_experts).
    mnkl = (m, n, k, g)
    ok = GroupedMixedInputGemmFp8Prefill.can_implement(
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

    # Resolve effective SFB granularity (defaults to SFA when omitted at the
    # runner level too).
    eff_sfb_granularity_k = (
        sfb_granularity_k if sfb_granularity_k is not None else sfa_granularity_k
    )

    moe_kernel = GroupedMixedInputGemmFp8Prefill(
        sfa_granularity_k,
        acc_dtype,
        mma_tiler_mnk,
        cluster_shape_mn,
        g,
        k,
        sfb_granularity_k=eff_sfb_granularity_k,
        use_pdl_rowsum=use_pdl_rowsum,
        use_simt_store=use_simt_store,
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
    if max_active_clusters_override is not None:
        max_active_clusters = max_active_clusters_override

    # Rowsum workspace is filled on the device immediately before GEMM.
    # BF16 SFB uses unweighted per-MMA-tile B rowsums; E8M0 uses rowsums
    # weighted by the same hardware SFB values consumed by blockscaled MMA.
    rowsum = build_rowsum_tensor(tensors.b, mma_tiler_mnk[2])

    compiled_kernel = cute.compile(
        moe_kernel,
        tensors.a.cute_tensor,
        tensors.a_scale.cute_tensor,
        tensors.b.cute_tensor,
        tensors.b_scale.cute_tensor,
        rowsum.cute_tensor,
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
            rowsum.cute_tensor,
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
            rowsum.cute_tensor,
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
        rowsum = build_rowsum_tensor(tensors.b, mma_tiler_mnk[2])
        return testing.JitArguments(
            tensors.a.cute_tensor,
            tensors.a_scale.cute_tensor,
            tensors.b.cute_tensor,
            tensors.b_scale.cute_tensor,
            rowsum.cute_tensor,
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
            + rowsum.ref_torch.numel() * rowsum.ref_torch.element_size()
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
        description="Contiguous grouped fp8 mixed-input GEMM runner with rowsum precompute."
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
        default=(128, 8, 256),
        help="Kernel tile shape (M_kernel, N_kernel, K_tile). Default 128,8,256.",
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
            "SFB activation-scale dtype. BFloat16 selects the post-MMA SFB "
            "policy; Float8E8M0FNU selects the MX HW-SFB policy."
        ),
    )

    parser.add_argument("--sfa_granularity_k", type=int, default=256)
    parser.add_argument(
        "--sfb_granularity_k",
        type=int,
        default=None,
        help=(
            "K-elements per SFB (activation) scale. Defaults to "
            "sfa_granularity_k when omitted. "
            "Must divide K. BF16 SFB requires a multiple of mma_tiler_mnk[2]; "
            "E8M0 SFB allows a multiple or divisor of mma_tiler_mnk[2] and "
            "requires a multiple of 32."
        ),
    )
    parser.add_argument("--tolerance", type=float, default=1e-01)
    parser.add_argument("--warmup_iterations", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument(
        "--non_uniform_group_sizes",
        action="store_true",
        help="If set, use random token group sizes instead of uniform N/G.",
    )
    parser.add_argument("--use_cold_l2", action="store_true")
    parser.add_argument(
        "--max_active_clusters",
        type=int,
        default=None,
        help="Override persistent scheduler grid.z.",
    )
    parser.add_argument(
        "--use_pdl_rowsum",
        action="store_true",
        help=(
            "Use Programmatic Dependent Launch between the rowsum precompute "
            "kernel and the main prefill GEMM."
        ),
    )
    parser.add_argument(
        "--use_simt_store",
        action="store_true",
        help="Diagnostic: use SIMT stores for all output tiles instead of TMA for full tiles.",
    )
    args = parser.parse_args()
    print(f"skip_ref_check={args.skip_ref_check}")
    print(f"use_pdl_rowsum={args.use_pdl_rowsum}")
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
        max_active_clusters_override=args.max_active_clusters,
        use_pdl_rowsum=args.use_pdl_rowsum,
        use_simt_store=args.use_simt_store,
    )
    print("PASS")
    print(f"{exec_time=}")
