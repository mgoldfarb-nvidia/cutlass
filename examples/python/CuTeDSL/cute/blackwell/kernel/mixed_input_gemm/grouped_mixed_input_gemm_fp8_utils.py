# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for the FP8 mixed-input grouped GEMM examples.

Keep this module limited to small CuTe layout/copy utilities and host-side
shape helpers. Kernel-specific pipeline policy should stay in the individual
kernel files.
"""

import argparse

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.mixed_input_helpers as mixed_input_utils
from cutlass.cute.nvgpu import tcgen05


def parse_comma_separated_ints(s: str) -> tuple[int, ...]:
    """Parse CLI comma-separated integer tuples."""
    try:
        return tuple(int(x.strip()) for x in s.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers."
        ) from exc


def check_contiguous_nb_alignment(
    dtype: type[cutlass.Numeric],
    contiguous_dim_size: int,
    expected_align_bytes: int,
) -> bool:
    """Return whether the contiguous M/N/B dimension satisfies byte alignment."""
    expected_alignment = expected_align_bytes * 8 // dtype.width
    return contiguous_dim_size % expected_alignment == 0


def validate_sfb_policy(
    sfb_dtype: type[cutlass.Numeric],
    sfb_granularity_k: int,
    default_sfb_granularity_k: int,
    mma_tiler: tuple[int, int, int],
    k: int,
) -> tuple[bool, int, bool]:
    """Validate SFB dtype and K granularity for BF16 and E8M0 policies."""
    if sfb_dtype is None:
        sfb_dtype = cutlass.BFloat16

    is_mxscale_sfb = sfb_dtype == cutlass.Float8E8M0FNU
    if sfb_dtype not in [cutlass.BFloat16, cutlass.Float8E8M0FNU]:
        print(f"Invalid sfb_dtype={sfb_dtype}")
        return False, default_sfb_granularity_k, is_mxscale_sfb

    bsbc_min_k = 32 * 4
    if is_mxscale_sfb and mma_tiler[2] < bsbc_min_k:
        print(
            f"Invalid mma_tiler K={mma_tiler[2]}: mxscale HW-SFB requires "
            f"K >= {bsbc_min_k}"
        )
        return False, default_sfb_granularity_k, is_mxscale_sfb

    eff_sfb_granularity_k = (
        sfb_granularity_k
        if sfb_granularity_k is not None
        else default_sfb_granularity_k
    )
    if is_mxscale_sfb:
        if (
            mma_tiler[2] % eff_sfb_granularity_k != 0
            and eff_sfb_granularity_k % mma_tiler[2] != 0
        ):
            print(
                f"Invalid sfb_granularity_k={eff_sfb_granularity_k} "
                f"not coalignable with mma_tiler[2]={mma_tiler[2]}"
            )
            return False, eff_sfb_granularity_k, is_mxscale_sfb
        if eff_sfb_granularity_k % 32 != 0:
            print(
                f"Invalid sfb_granularity_k={eff_sfb_granularity_k}: "
                "mxscale HW-SFB requires a multiple of 32"
            )
            return False, eff_sfb_granularity_k, is_mxscale_sfb
    elif eff_sfb_granularity_k % mma_tiler[2] != 0:
        print(
            f"Invalid sfb_granularity_k={eff_sfb_granularity_k} "
            f"not a multiple of mma_tiler[2]={mma_tiler[2]}"
        )
        return False, eff_sfb_granularity_k, is_mxscale_sfb

    if k % eff_sfb_granularity_k != 0:
        print(
            f"Invalid sfb_granularity_k={eff_sfb_granularity_k} not a divisor of K={k}"
        )
        return False, eff_sfb_granularity_k, is_mxscale_sfb

    return True, eff_sfb_granularity_k, is_mxscale_sfb


def aligned_smem_bytes(
    dtype: type[cutlass.Numeric],
    layout: cute.Layout,
    alignment: int = 1024,
) -> int:
    """Return SMEM bytes for ``layout`` rounded to the allocator alignment."""
    return cute.round_up(cute.size_in_bytes(dtype, layout), alignment)


def blockscaled_scale_tmem_cols(cta_tile_k: int, sf_vec_size: int) -> int:
    """Return scale-factor TMEM columns, padding sub-BSBC K tiles to one atom."""
    bsbc_min_k = sf_vec_size * 4
    sf_k = max(cta_tile_k, bsbc_min_k)
    return cute.round_up(sf_k // bsbc_min_k, 4)


def divide_tensor_by_tiler(
    tensor: cute.Tensor,
    transform_tiler: cute.Layout,
) -> cute.Tensor:
    """Flat-divide a tensor by a tiler and group rest modes into one mode."""
    divided_tensor = cute.flat_divide(tensor, transform_tiler)
    return cute.group_modes(divided_tensor, 1, cute.rank(divided_tensor))


@cute.jit
def make_single_element_tensor_view(tensor: cute.Tensor, coord) -> cute.Tensor:
    """Return a one-element tensor view rooted at ``tensor[coord]``."""
    return cute.make_tensor(
        tensor.iterator + tensor.layout(coord),
        cute.make_layout(1),
    )


@cute.jit
def stage_bf16_sfb_gmem_to_smem(
    mSFB_nkl: cute.Tensor,
    sSFB_post: cute.Tensor,
    cpasync_atom,
    epi_tidx: cutlass.Int32,
    coord_n: cutlass.Int32,
    distance_to_boundary: cutlass.Int32,
    num_threads: cutlass.Constexpr[int],
    sfb_num_chunks: cutlass.Constexpr[int],
    sfb_n: cutlass.Constexpr[int],
    zero_fill_oob: cutlass.Constexpr[bool],
) -> cutlass.Int32:
    """Stage BF16 SFB to token-major SMEM and return the row offset to read."""
    elems_per_int32: cutlass.Constexpr[int] = 32 // cutlass.BFloat16.width
    sfb_smem_row_offset = cutlass.Int32(0)
    if (distance_to_boundary >= sfb_n) and ((coord_n % elems_per_int32) == 0):
        n_int32_per_chunk: cutlass.Constexpr[int] = sfb_n // elems_per_int32
        sfb_total_int32: cutlass.Constexpr[int] = n_int32_per_chunk * sfb_num_chunks
        per_thr_i32_loads: cutlass.Constexpr[int] = (
            sfb_total_int32 + num_threads - 1
        ) // num_threads
        mSFB_nkl_i32 = cute.recast_tensor(mSFB_nkl, cutlass.Int32)
        sSFB_post_i32 = cute.recast_tensor(sSFB_post, cutlass.Int32)
        for li in cutlass.range_constexpr(per_thr_i32_loads):
            idx = epi_tidx + li * num_threads
            if idx < sfb_total_int32:
                chunk_idx = idx // n_int32_per_chunk
                n_int32_local = idx % n_int32_per_chunk
                n_int32_global = (coord_n // elems_per_int32) + n_int32_local
                src_view = make_single_element_tensor_view(
                    mSFB_nkl_i32, (n_int32_global, chunk_idx, 0)
                )
                dst_view = make_single_element_tensor_view(
                    sSFB_post_i32, (n_int32_local, chunk_idx)
                )
                cute.copy(cpasync_atom, src_view, dst_view)
    elif (distance_to_boundary >= sfb_n) and ((coord_n + sfb_n) < mSFB_nkl.shape[0]):
        sfb_smem_row_offset = cutlass.Int32(1)
        n_shifted_int32_per_chunk: cutlass.Constexpr[int] = (
            sfb_n + elems_per_int32
        ) // elems_per_int32
        sfb_shifted_total_int32: cutlass.Constexpr[int] = (
            n_shifted_int32_per_chunk * sfb_num_chunks
        )
        per_thr_shifted_loads: cutlass.Constexpr[int] = (
            sfb_shifted_total_int32 + num_threads - 1
        ) // num_threads
        mSFB_nkl_i32 = cute.recast_tensor(mSFB_nkl, cutlass.Int32)
        sSFB_post_i32 = cute.recast_tensor(sSFB_post, cutlass.Int32)
        sfb_shifted_start_i32 = (coord_n - cutlass.Int32(1)) // elems_per_int32
        for li in cutlass.range_constexpr(per_thr_shifted_loads):
            idx = epi_tidx + li * num_threads
            if idx < sfb_shifted_total_int32:
                chunk_idx = idx // n_shifted_int32_per_chunk
                n_int32_local = idx % n_shifted_int32_per_chunk
                n_int32_global = sfb_shifted_start_i32 + n_int32_local
                src_view = make_single_element_tensor_view(
                    mSFB_nkl_i32, (n_int32_global, chunk_idx, 0)
                )
                dst_view = make_single_element_tensor_view(
                    sSFB_post_i32, (n_int32_local, chunk_idx)
                )
                cute.copy(cpasync_atom, src_view, dst_view)
    else:
        sfb_total: cutlass.Constexpr[int] = sfb_n * sfb_num_chunks
        per_thr_loads: cutlass.Constexpr[int] = (
            sfb_total + num_threads - 1
        ) // num_threads
        for li in cutlass.range_constexpr(per_thr_loads):
            idx = epi_tidx + li * num_threads
            if idx < sfb_total:
                chunk_idx = idx // sfb_n
                n_local = idx % sfb_n
                n_global = coord_n + n_local
                if (n_local < distance_to_boundary) and (n_global < mSFB_nkl.shape[0]):
                    sSFB_post[(n_local, chunk_idx)] = mSFB_nkl[(n_global, chunk_idx, 0)]
                elif cutlass.const_expr(zero_fill_oob):
                    sSFB_post[(n_local, chunk_idx)] = cutlass.BFloat16(0.0)
    return sfb_smem_row_offset


@cute.jit
def store_accumulator_subtiles(
    tTR_rAcc_final: cute.Tensor,
    tTR_rC: cute.Tensor,
    tTR_gC: cute.Tensor,
    tiled_copy_r2s,
    tRS_rC: cute.Tensor,
    tRS_sC: cute.Tensor,
    tma_atom_c,
    bSG_sC: cute.Tensor,
    bSG_gC: cute.Tensor,
    simt_atom,
    c_pipeline,
    epilog_sync_barrier,
    m_thr_offset: cute.Tensor,
    warp_idx: cutlass.Int32,
    tma_store_warp_id: cutlass.Constexpr[int],
    cta_coord_m: cutlass.Int32,
    distance_to_boundary: cutlass.Int32,
    c_shape_m: cutlass.Int32,
    cta_tile_m: cutlass.Constexpr[int],
    cta_tile_n: cutlass.Constexpr[int],
    num_c_stage: cutlass.Constexpr[int],
    c_dtype: type[cutlass.Numeric],
    num_prev_subtiles: cutlass.Int32,
    force_simt_store: cutlass.Constexpr[bool],
) -> cutlass.Int32:
    """Store one epilogue accumulator tile.

    Default policy uses TMA stores for complete token tiles and predicated
    SIMT stores for ragged expert tails. ``force_simt_store`` supports the
    MX prefill all-SIMT tuning knob without duplicating the store loop.
    """
    subtile_cnt = cute.size(tTR_rAcc_final.shape, mode=[3])
    for subtile_idx in cutlass.range(subtile_cnt):
        tTR_rAcc_subtile = tTR_rAcc_final[(None, None, None, subtile_idx)]
        if cutlass.const_expr(force_simt_store):
            acc_vec_simt = tTR_rAcc_subtile.load()
            acc_vec_simt = acc_vec_simt.to(c_dtype)
            tTR_rC.store(acc_vec_simt)
            if distance_to_boundary >= cta_tile_n:
                cute.copy(
                    simt_atom,
                    cute.flatten(tTR_rC),
                    cute.flatten(tTR_gC[(None, None, None, subtile_idx)]),
                )
            else:
                tCpC = cute.make_rmem_tensor(
                    cute.make_layout(tTR_rC.shape),
                    cutlass.Boolean,
                )
                m_thr_slice = m_thr_offset[(None, None, None, subtile_idx)]
                for i in cutlass.range(cute.size(tCpC), unroll_full=True):
                    tCpC[i] = (
                        m_thr_slice[(i)][0] + cta_coord_m * cta_tile_m < c_shape_m
                    ) and (m_thr_slice[(i)][1] < distance_to_boundary)
                cute.copy(
                    simt_atom,
                    cute.flatten(tTR_rC),
                    cute.flatten(tTR_gC[(None, None, None, subtile_idx)]),
                    pred=cute.flatten(tCpC),
                )
        elif distance_to_boundary >= cta_tile_n:
            acc_vec_tma = tiled_copy_r2s.retile(tTR_rAcc_subtile).load()
            acc_vec_tma = acc_vec_tma.to(c_dtype)
            tRS_rC.store(acc_vec_tma)
            num_prev_subtiles += 1
            c_buffer = num_prev_subtiles % num_c_stage
            cute.copy(
                tiled_copy_r2s,
                tRS_rC,
                tRS_sC[(None, None, None, c_buffer)],
            )
            cute.arch.fence_proxy("async.shared", space="cta")
            epilog_sync_barrier.arrive_and_wait()
            if warp_idx == tma_store_warp_id:
                cute.copy(
                    tma_atom_c,
                    bSG_sC[(None, c_buffer)],
                    bSG_gC[(None, subtile_idx)],
                )
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()
            epilog_sync_barrier.arrive_and_wait()
        else:
            acc_vec_tail = tTR_rAcc_subtile.load()
            acc_vec_tail = acc_vec_tail.to(c_dtype)
            tTR_rC.store(acc_vec_tail)
            tCpC = cute.make_rmem_tensor(
                cute.make_layout(tTR_rC.shape),
                cutlass.Boolean,
            )
            m_thr_slice = m_thr_offset[(None, None, None, subtile_idx)]
            for i in cutlass.range(cute.size(tCpC), unroll_full=True):
                tCpC[i] = (
                    m_thr_slice[(i)][0] + cta_coord_m * cta_tile_m < c_shape_m
                ) and (m_thr_slice[(i)][1] < distance_to_boundary)
            cute.copy(
                simt_atom,
                cute.flatten(tTR_rC),
                cute.flatten(tTR_gC[(None, None, None, subtile_idx)]),
                pred=cute.flatten(tCpC),
            )
    return num_prev_subtiles


@cute.jit
def store_prefill_accumulator_tile(
    tTR_rAcc_final: cute.Tensor,
    tTR_rC: cute.Tensor,
    tTR_gC_partitioned: cute.Tensor,
    tiled_copy_r2s,
    tRS_rC: cute.Tensor,
    tRS_sC: cute.Tensor,
    tma_atom_c,
    bSG_sC: cute.Tensor,
    bSG_gC_partitioned: cute.Tensor,
    simt_atom,
    c_pipeline,
    epilog_sync_barrier,
    m_thr_offset: cute.Tensor,
    warp_idx: cutlass.Int32,
    tma_store_warp_id: cutlass.Constexpr[int],
    cta_coord_m: cutlass.Int32,
    coord_n: cutlass.Int32,
    distance_to_boundary: cutlass.Int32,
    c_shape_m: cutlass.Int32,
    c_stride_n,
    cta_tile_m: cutlass.Constexpr[int],
    cta_tile_n: cutlass.Constexpr[int],
    num_c_stage: cutlass.Constexpr[int],
    c_dtype: type[cutlass.Numeric],
    num_prev_subtiles: cutlass.Int32,
    force_simt_store: cutlass.Constexpr[bool],
    is_n_major_c: cutlass.Constexpr[bool],
    cta_v_size: cutlass.Constexpr[int],
) -> cutlass.Int32:
    """Store one prefill accumulator tile and protect single-stage epilog scratch."""
    loop_m_tile = cta_coord_m // cta_v_size

    bSG_gC = bSG_gC_partitioned[(None, None, None, loop_m_tile, 0, 0)]
    tma_store_offset_coord = (
        (coord_n, 0, 0) if cutlass.const_expr(is_n_major_c) else (0, coord_n, 0)
    )
    bSG_gC = cute.make_tensor(
        (
            tma_store_offset_coord[0] + bSG_gC.iterator[0],
            tma_store_offset_coord[1] + bSG_gC.iterator[1],
            tma_store_offset_coord[2] + bSG_gC.iterator[2],
        ),
        bSG_gC.layout,
    )
    tTR_gC = tTR_gC_partitioned[(None, None, None, None, None, loop_m_tile, 0, 0)]
    tTR_gC = cute.make_tensor(
        tTR_gC.iterator + (coord_n * c_stride_n),
        tTR_gC.layout,
    )
    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
    tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))

    num_prev_subtiles = store_accumulator_subtiles(
        tTR_rAcc_final,
        tTR_rC,
        tTR_gC,
        tiled_copy_r2s,
        tRS_rC,
        tRS_sC,
        tma_atom_c,
        bSG_sC,
        bSG_gC,
        simt_atom,
        c_pipeline,
        epilog_sync_barrier,
        m_thr_offset,
        warp_idx,
        tma_store_warp_id,
        cta_coord_m,
        distance_to_boundary,
        c_shape_m,
        cta_tile_m,
        cta_tile_n,
        num_c_stage,
        c_dtype,
        num_prev_subtiles,
        force_simt_store,
    )

    # Full-tile TMA stores synchronize epilogue threads inside the helper above;
    # ragged SIMT tails and forced SIMT stores do not. The prefill epilog uses
    # single-stage rowsum/SFB scratch, so all epilogue threads must rendezvous
    # before the next work tile can overwrite that scratch.
    if cutlass.const_expr(force_simt_store):
        epilog_sync_barrier.arrive_and_wait()
    elif distance_to_boundary < cta_tile_n:
        epilog_sync_barrier.arrive_and_wait()

    return num_prev_subtiles


@cute.jit
def store_decode_accumulator_tiles(
    tTR_rAcc_final_tile0: cute.Tensor,
    tTR_rAcc_final_tile1: cute.Tensor,
    tTR_rC: cute.Tensor,
    tTR_gC_partitioned: cute.Tensor,
    tiled_copy_r2s,
    tRS_rC: cute.Tensor,
    tRS_sC: cute.Tensor,
    tma_atom_c,
    bSG_sC: cute.Tensor,
    bSG_gC_partitioned: cute.Tensor,
    simt_atom,
    c_pipeline,
    epilog_sync_barrier,
    m_thr_offset: cute.Tensor,
    warp_idx: cutlass.Int32,
    tma_store_warp_id: cutlass.Constexpr[int],
    cta_coord_m: cutlass.Int32,
    coord_n: cutlass.Int32,
    distance_to_boundary: cutlass.Int32,
    c_shape_m: cutlass.Int32,
    c_stride_n,
    cta_tile_m: cutlass.Constexpr[int],
    cta_tile_n: cutlass.Constexpr[int],
    num_c_stage: cutlass.Constexpr[int],
    c_dtype: type[cutlass.Numeric],
    num_prev_subtiles: cutlass.Int32,
    is_n_major_c: cutlass.Constexpr[bool],
) -> cutlass.Int32:
    """Store decode epilogue accumulators for the two local M tiles."""
    loop_m_base = cta_coord_m * 2
    for ni_tile in cutlass.range_constexpr(2):
        tTR_rAcc_final_sel = (
            tTR_rAcc_final_tile0 if ni_tile == 0 else tTR_rAcc_final_tile1
        )
        loop_m_tile = loop_m_base + ni_tile

        bSG_gC = bSG_gC_partitioned[(None, None, None, loop_m_tile, 0, 0)]
        tma_store_offset_coord = (
            (coord_n, 0, 0) if cutlass.const_expr(is_n_major_c) else (0, coord_n, 0)
        )
        bSG_gC = cute.make_tensor(
            (
                tma_store_offset_coord[0] + bSG_gC.iterator[0],
                tma_store_offset_coord[1] + bSG_gC.iterator[1],
                tma_store_offset_coord[2] + bSG_gC.iterator[2],
            ),
            bSG_gC.layout,
        )

        tTR_gC = tTR_gC_partitioned[(None, None, None, None, None, loop_m_tile, 0, 0)]
        tTR_gC = cute.make_tensor(
            tTR_gC.iterator + (coord_n * c_stride_n),
            tTR_gC.layout,
        )
        bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
        tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))

        num_prev_subtiles = store_accumulator_subtiles(
            tTR_rAcc_final_sel,
            tTR_rC,
            tTR_gC,
            tiled_copy_r2s,
            tRS_rC,
            tRS_sC,
            tma_atom_c,
            bSG_sC,
            bSG_gC,
            simt_atom,
            c_pipeline,
            epilog_sync_barrier,
            m_thr_offset,
            warp_idx,
            tma_store_warp_id,
            loop_m_tile,
            distance_to_boundary,
            c_shape_m,
            cta_tile_m,
            cta_tile_n,
            num_c_stage,
            c_dtype,
            num_prev_subtiles,
            False,
        )

    return num_prev_subtiles


def epilog_and_acc_update_tmem_copy_and_partition(
    tidx: cutlass.Int32,
    tAcc: cute.Tensor,
    gC_mnl: cute.Tensor,
    scale_tensor: cute.Tensor,
    epi_tile: cute.Tile,
    cta_tile_shape_mnk: tuple[int, int, int],
    c_layout: utils.LayoutEnum,
    c_dtype: type[cutlass.Numeric],
    acc_dtype: type[cutlass.Numeric],
    use_2cta_instrs: bool,
) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor]:
    """Partition accumulator, C, and scale tensors for epilog T2R work."""
    copy_atom_t2r = sm100_utils.get_tmem_load_op(
        cta_tile_shape_mnk,
        c_layout,
        c_dtype,
        acc_dtype,
        epi_tile,
        use_2cta_instrs,
    )
    tAcc_epi = cute.flat_divide(
        tAcc[((None, None), 0, 0, None)],
        epi_tile,
    )
    tiled_copy_t2r = tcgen05.make_tmem_copy(
        copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
    )

    thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
    tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

    gC_mnl_epi = cute.flat_divide(
        gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
    )
    sScale_epi = cute.flat_divide(scale_tensor, epi_tile)
    tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
    tTR_sScale = thr_copy_t2r.partition_D(sScale_epi)
    tTR_rAcc = cute.make_rmem_tensor(
        tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, acc_dtype
    )
    tTR_rAcc_final_ = cute.make_rmem_tensor(
        tTR_gC[(None, None, None, None, None, 0, 0, 0)].shape, acc_dtype
    )
    tTR_rAcc_final = cute.group_modes(tTR_rAcc_final_, 3, cute.rank(tTR_rAcc_final_))
    return (
        tiled_copy_t2r,
        tTR_tAcc,
        tTR_rAcc,
        tTR_rAcc_final,
        tTR_sScale,
    )


def mainloop_s2t_copy_and_partition(
    sSF: cute.Tensor,
    tSF: cute.Tensor,
    sf_mma_dtype: type[cutlass.Numeric],
    cta_group,
) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
    """Build and partition the SMEM-to-TMEM scale-factor copy."""
    tCsSF_compact = cute.filter_zeros(sSF)
    tCtSF_compact = cute.filter_zeros(tSF)

    copy_atom_s2t = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(cta_group),
        sf_mma_dtype,
    )
    tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
    thr_copy_s2t = tiled_copy_s2t.get_slice(0)

    tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
    tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
        tiled_copy_s2t, tCsSF_compact_s2t_
    )
    tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

    return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t


def compute_persistent_grid(
    c: cute.Tensor,
    cta_tile_shape_mnk: tuple[int, int, int],
    cluster_shape_mn: tuple[int, int],
    max_active_clusters: cutlass.Constexpr,
    m_tile_multiplier: int = 1,
) -> tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]:
    """Compute persistent scheduler params and launch grid for C tiles."""
    if m_tile_multiplier == 1:
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
    else:
        c_shape = (cta_tile_shape_mnk[0] * m_tile_multiplier, cta_tile_shape_mnk[1])
    gc = cute.zipped_divide(c, tiler=c_shape)
    num_ctas_mnl = gc[(0, (None, None, None))].shape
    cluster_shape_mnl = (*cluster_shape_mn, 1)

    tile_sched_params = utils.PersistentTileSchedulerParams(
        num_ctas_mnl, cluster_shape_mnl
    )
    grid = (cluster_shape_mn[0], cluster_shape_mn[1], max_active_clusters)
    return tile_sched_params, grid


@cute.jit
def produce_grouped_tile_info(
    tile_sched_params: utils.PersistentTileSchedulerParams,
    bidx: cutlass.Int32,
    bidy: cutlass.Int32,
    bidz: cutlass.Int32,
    block_in_cluster_coord_vmnk,
    tile_info_pipeline,
    sTile_info: cute.Tensor,
    sched_sync_barrier,
    num_tile_info_stage: cutlass.Constexpr[int],
    cluster_tile_shape_mnk: tuple[int, int, int],
    cluster_shape_mn: tuple[int, int],
    cta_tile_shape_mnk: tuple[int, int, int],
    group_count: cutlass.Int32,
    cumsum: cute.Tensor,
) -> None:
    """Produce ragged-group work-tile metadata for tile-info consumers."""
    tile_sched = utils.StaticPersistentRuntimeTileScheduler.create(
        tile_sched_params,
        (bidx, bidy, bidz),
        cute.arch.grid_dim(),
        inner_mode=0,
    )
    work_tile = tile_sched.initial_work_tile_info()
    tile_info_producer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, num_tile_info_stage
    )
    search_state = mixed_input_utils.create_initial_contiguous_group_search_state()
    not_last_tile = cutlass.Boolean(1)
    while not_last_tile:
        tile_info_pipeline.producer_acquire(tile_info_producer_state)
        cluster_tile_coord_mnl = work_tile.tile_idx
        cta_tile_coord_m = (
            cluster_tile_coord_mnl[0] * cluster_shape_mn[0]
            + block_in_cluster_coord_vmnk[1] * 1
            + block_in_cluster_coord_vmnk[0]
        )
        cta_tile_offset_n = block_in_cluster_coord_vmnk[2]
        search_state = mixed_input_utils.contiguous_group_search(
            cluster_tile_shape_mnk,
            group_count,
            cluster_tile_coord_mnl[1],
            search_state,
            cumsum,
            1,
        )
        cur_sTile_info = sTile_info[(None, tile_info_producer_state.index)]
        not_last_tile = search_state.cur_group_idx <= group_count
        with cute.arch.elect_one():
            cur_sTile_info[0] = cta_tile_coord_m
            cur_sTile_info[1] = (
                search_state.cur_start + cta_tile_offset_n * cta_tile_shape_mnk[1]
            )
            cur_sTile_info[2] = search_state.cur_group_idx - 1
            cur_sTile_info[3] = (
                search_state.cur_boundary
                - search_state.cur_start
                - (cta_tile_offset_n * cta_tile_shape_mnk[1])
            )
        cute.arch.fence_proxy("async.shared", space="cta")
        sched_sync_barrier.arrive_and_wait()
        tile_info_pipeline.producer_commit(tile_info_producer_state)
        tile_info_producer_state.advance()
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    tile_info_pipeline.producer_tail(tile_info_producer_state)
