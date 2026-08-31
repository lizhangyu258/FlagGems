# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
from pathlib import Path

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

logger = logging.getLogger(__name__)

_TLE_RAW_SOURCE = Path(__file__).with_name("sort_tle_raw.cu")


def _configure_tle_raw_cccl_include():
    if device.vendor_name != "nvidia":
        return
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    cccl_include = cuda_home / "include" / "cccl"
    if not (cccl_include / "cub").is_dir():
        return
    try:
        from triton import knobs
    except (AttributeError, ImportError):
        return
    try:
        flags = knobs.nvidia.tle_raw_clang_flags or ""
    except AttributeError:
        return

    required_flags = [f"-I{cccl_include}"]
    try:
        cuda_version = json.loads((cuda_home / "version.json").read_text())
        cuda_major = int(cuda_version["cuda"]["version"].split(".", 1)[0])
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        cuda_major = 0
    if cuda_major >= 13:
        # Clang's CUDA wrapper suppresses the header defining this CUDA 13
        # macro but still includes math_functions.hpp.
        required_flags.append("-D_NV_RSQRT_SPECIFIER=")

    existing_flags = flags.split()
    missing_flags = [flag for flag in required_flags if flag not in existing_flags]
    if missing_flags:
        knobs.nvidia.tle_raw_clang_flags = " ".join((flags, *missing_flags)).strip()


_configure_tle_raw_cccl_include()

if device.vendor_name == "nvidia":
    try:
        import triton.experimental.tle.language.gpu as tle_gpu
        import triton.experimental.tle.language.raw as tle_raw
        from triton.experimental.tle.raw import dialect

        _HAS_TLE_RAW_SORT = _TLE_RAW_SOURCE.is_file()
    except (AttributeError, ImportError, ModuleNotFoundError):
        tle_gpu = None
        tle_raw = None
        dialect = None
        _HAS_TLE_RAW_SORT = False
else:
    tle_gpu = None
    tle_raw = None
    dialect = None
    _HAS_TLE_RAW_SORT = False


if _HAS_TLE_RAW_SORT:

    @dialect(
        name="cuda",
        file=_TLE_RAW_SOURCE,
        extern_func_name="RadixHistogramDigits8x2048",
        deferred=True,
    )
    def _radix_histogram_digits_8x2048_raw(*args, **kwargs): ...

    @dialect(
        name="cuda",
        file=_TLE_RAW_SOURCE,
        extern_func_name="RadixRank8x2048Precomputed",
        deferred=True,
    )
    def _radix_rank_8x2048_precomputed_raw(*args, **kwargs): ...

    @triton.jit
    def _ordered_float16_key(value, descending: tl.constexpr, dtype_kind: tl.constexpr):
        bits = value.to(tl.uint16, bitcast=True)
        sign = tl.full((), 0x8000, tl.uint16)
        magnitude = bits & tl.full((), 0x7FFF, tl.uint16)
        infinity: tl.constexpr = 0x7C00 if dtype_kind == 0 else 0x7F80
        # Match torch.sort semantics: +/-0 compare equal and all NaNs form one
        # stable group at the end (ascending) or beginning (descending).
        normalized = tl.where(magnitude == 0, 0, bits)
        ordered = tl.where((normalized & sign) != 0, ~normalized, normalized ^ sign)
        ordered = ~ordered if descending else ordered
        nan_key: tl.constexpr = 0 if descending else 0xFFFF
        return tl.where(magnitude > infinity, nan_key, ordered).to(tl.uint16)

    @triton.jit
    def _radix_tile_histogram_kernel_raw(
        input_ptr,
        counts_ptr,
        rows,
        n,
        descending: tl.constexpr,
        dtype_kind: tl.constexpr,
        BIT_OFFSET: tl.constexpr,
        TILE_N: tl.constexpr,
    ):
        tl.static_assert(TILE_N == 2048)
        program = tl.program_id(0)
        tiles = tl.cdiv(n, TILE_N)
        row = program // tiles
        tile = program - row * tiles
        columns = tile * TILE_N + tl.arange(0, TILE_N)
        mask = columns < n
        values = tl.load(input_ptr + row * n + columns, mask=mask)
        keys = _ordered_float16_key(values, descending, dtype_kind)
        digits = ((keys >> BIT_OFFSET) & 0xFF).to(tl.uint16)

        digits_smem = tle_gpu.alloc(
            shape=[TILE_N],
            dtype=tl.uint16,
            layout=None,
            scope=tle_gpu.smem,
            nv_mma_shared_layout=False,
        )
        counts_smem = tle_gpu.alloc(
            shape=[256],
            dtype=tl.int32,
            layout=None,
            scope=tle_gpu.smem,
            nv_mma_shared_layout=False,
        )
        tl.store(tle_gpu.local_ptr(digits_smem, (columns,)), digits)
        valid_count = tl.minimum(TILE_N, n - tile * TILE_N)
        counts_smem = tle_raw.call_smem(
            _radix_histogram_digits_8x2048_raw,
            [digits_smem, counts_smem, valid_count],
            output_indices=[1],
        )

        bins = tl.arange(0, 256)
        counts = tl.load(tle_gpu.local_ptr(counts_smem, (bins,)))
        tl.store(counts_ptr + (row * tiles + tile) * 256 + bins, counts)

    @triton.jit
    def _radix_tile_offsets_kernel(counts_ptr, offsets_ptr, tiles):
        row = tl.program_id(0)
        bins = tl.arange(0, 256)
        row_base = row * tiles * 256

        bin_totals = tl.zeros((256,), dtype=tl.int32)
        for tile in range(0, tiles):
            counts = tl.load(counts_ptr + row_base + tile * 256 + bins)
            bin_totals += counts

        bin_bases = tl.cumsum(bin_totals, axis=0) - bin_totals
        running = bin_bases
        for tile in range(0, tiles):
            offset = row_base + tile * 256 + bins
            counts = tl.load(counts_ptr + offset)
            tl.store(offsets_ptr + offset, running)
            running += counts

    @triton.jit
    def _sweep_cub_local_rank_precomputed(
        arr_ptr,
        associate_arr_ptr,
        out_ptr,
        associate_out_ptr32,
        associate_out_ptr64,
        tile_offsets_ptr,
        bit_offset,
        N,
        OUT_N,
        TILE_N: tl.constexpr,
        TILE_R: tl.constexpr,
        k_bits: tl.constexpr,
        descending: tl.constexpr,
        dtype_kind: tl.constexpr,
        final_pass,
    ):
        tl.static_assert(TILE_N == 2048)
        tl.static_assert(TILE_R == 256)
        tl.static_assert(k_bits == 8)

        pid = tl.program_id(0)
        pid_m = pid // OUT_N
        pid_n = pid - pid_m * OUT_N
        cols = tl.arange(0, TILE_N)
        n_offsets = pid_n * TILE_N + cols
        mask = n_offsets < N
        arr = tl.load(arr_ptr + pid_m * N + n_offsets, mask=mask)
        arr_u = _ordered_float16_key(arr, descending, dtype_kind)
        digits = ((arr_u >> bit_offset) & 0xFF).to(tl.uint16)
        digits = tl.where(mask, digits, 0xFF).to(tl.uint16)

        digits_smem = tle_gpu.alloc(
            shape=[TILE_N],
            dtype=tl.uint16,
            layout=None,
            scope=tle_gpu.smem,
            nv_mma_shared_layout=False,
        )
        tl.store(tle_gpu.local_ptr(digits_smem, (cols,)), digits)
        valid_count = tl.minimum(TILE_N, N - pid_n * TILE_N)
        tle_raw.call_smem(
            _radix_rank_8x2048_precomputed_raw,
            [
                digits_smem,
                arr_ptr,
                associate_arr_ptr,
                out_ptr,
                associate_out_ptr32,
                associate_out_ptr64,
                tile_offsets_ptr,
                pid_m,
                pid_n,
                N,
                OUT_N,
                valid_count,
                final_pass,
            ],
            output_indices=[],
        )


def _tle_raw_dtype_kind(dtype):
    if dtype == torch.float16:
        return 0
    if dtype == torch.bfloat16:
        return 1
    raise TypeError(f"TLE Raw radix sort does not support {dtype}")


def _can_use_tle_raw_sort(inp, dim):
    if (
        not _HAS_TLE_RAW_SORT
        or inp.ndim == 0
        or inp.numel() >= 1 << 31
        or inp.dtype not in (torch.float16, torch.bfloat16)
    ):
        return False
    normalized_dim = dim + inp.ndim if dim < 0 else dim
    if normalized_dim < 0 or normalized_dim >= inp.ndim:
        return False
    n = inp.shape[normalized_dim]
    return 1 < n < (1 << 30)


def radix_sort(arr, descending=False):
    n = arr.shape[-1]
    assert n < (1 << 30), "we have not implemented 2**30 per launch"
    dtype = arr.dtype
    if dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"radix_sort only supports float16/bfloat16, got {dtype}")
    if not _HAS_TLE_RAW_SORT:
        raise RuntimeError("radix_sort requires NVIDIA TLE Raw support")

    rows = arr.numel() // n
    tile_n = 2048
    radix_bins = 256
    tiles = triton.cdiv(n, tile_n)
    tile_grid = (rows * tiles,)
    sweep_grid = (rows * tiles, 1)
    dtype_kind = _tle_raw_dtype_kind(dtype)

    with torch_device_fn.device(arr.device):
        arr_in = torch.clone(arr)
        arr_out = torch.empty_like(arr)
        temporary_indices = torch.empty_like(arr, dtype=torch.int32)
        final_indices = torch.empty_like(arr, dtype=torch.int64)
        tile_counts = torch.empty(
            (rows, tiles, radix_bins), device=arr.device, dtype=torch.int32
        )
        tile_offsets = torch.empty_like(tile_counts)

        for pass_id in range(2):
            bit_offset = pass_id * 8
            _radix_tile_histogram_kernel_raw[tile_grid](
                arr_in,
                tile_counts,
                rows,
                n,
                descending=descending,
                dtype_kind=dtype_kind,
                BIT_OFFSET=bit_offset,
                TILE_N=tile_n,
                num_warps=8,
            )
            _radix_tile_offsets_kernel[(rows,)](
                tile_counts,
                tile_offsets,
                tiles,
                num_warps=8,
            )
            _sweep_cub_local_rank_precomputed[sweep_grid](
                arr_in,
                temporary_indices,
                arr_out,
                temporary_indices,
                final_indices,
                tile_offsets,
                bit_offset,
                n,
                tiles,
                tile_n,
                radix_bins,
                8,
                int(descending),
                dtype_kind,
                int(pass_id == 1),
                num_warps=8,
            )
            arr_in, arr_out = arr_out, arr_in

    return arr_in, final_indices


def sort(inp, dim=-1, descending=False):
    logger.debug("GEMS SORT")
    return sort_stable(inp, stable=False, dim=dim, descending=descending)


def sort_stable(inp, *, stable, dim=-1, descending=False):
    logger.debug("GEMS SORT.STABLE")
    _ = stable
    if inp.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            f"sort currently only supports float16/bfloat16, got {inp.dtype}"
        )

    sort_elem_cnt = inp.shape[dim]
    if sort_elem_cnt == 0:
        return torch.empty_like(inp), torch.empty_like(inp, dtype=torch.int64)
    if sort_elem_cnt == 1:
        return inp, torch.zeros_like(inp, dtype=torch.int64)

    if dim < 0:
        dim += inp.ndim
    if dim != inp.ndim - 1:
        inp = torch.movedim(inp, dim, -1).contiguous()
    else:
        inp = inp.contiguous()

    out, out_index = radix_sort(inp, descending)

    if dim != inp.ndim - 1:
        out = torch.movedim(out, -1, dim)
        out_index = torch.movedim(out_index, -1, dim)
    return out, out_index
