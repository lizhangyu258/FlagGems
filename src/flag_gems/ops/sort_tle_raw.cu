// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdint.h>

#define GLOBAL __attribute__((address_space(1)))

namespace {

constexpr int kRadixSize = 256;
constexpr int kThreads = 256;
constexpr int kWarps = 8;
constexpr int kLargeTile = 2048;
constexpr int kSmallCapacity = 4096;

static_assert(kThreads == kWarps * 32, "radix rank assumes eight full warps");
static_assert(
    2 * kSmallCapacity * sizeof(uint16_t) +
            2 * kSmallCapacity * sizeof(uint16_t) +
            kSmallCapacity * sizeof(uint16_t) +
            2 * kRadixSize * sizeof(uint32_t) +
            kWarps * kRadixSize * sizeof(uint16_t) <=
        48 * 1024,
    "small sort must stay within the portable static shared-memory limit");

constexpr unsigned long long kAggregate = 1ULL << 62;
constexpr unsigned long long kInclusive = 2ULL << 62;
constexpr unsigned long long kValueMask = (1ULL << 62) - 1;

__device__ __forceinline__ unsigned lane_mask_lt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  return mask;
}

// Convert IEEE fp16/bf16 bits to an unsigned key whose natural order matches
// torch.sort.  Equal zeros and all NaNs deliberately share a key so that the
// stable radix passes preserve their input order and their original bit pattern.
__device__ __forceinline__ uint16_t ordered_key(uint16_t bits, int dtype_kind,
                                                int descending) {
  const uint16_t abs_bits = bits & 0x7fffU;
  const uint16_t exp_mask = dtype_kind == 0 ? 0x7c00U : 0x7f80U;
  const uint16_t mantissa_mask = dtype_kind == 0 ? 0x03ffU : 0x007fU;
  const bool is_nan = (abs_bits & exp_mask) == exp_mask &&
                      (abs_bits & mantissa_mask) != 0;

  uint16_t key;
  if (is_nan) {
    key = 0xffffU;
  } else {
    // Canonicalize signed zero for comparison only.  The value moved by the
    // sorter remains the original two bytes.
    const uint16_t comparable = abs_bits == 0 ? 0 : bits;
    key = (comparable & 0x8000U) != 0 ? static_cast<uint16_t>(~comparable)
                                      : static_cast<uint16_t>(comparable ^ 0x8000U);
  }
  return descending ? static_cast<uint16_t>(~key) : key;
}

template <int Capacity>
__device__ __forceinline__ void stable_digit_ranks(
    uint16_t *values, uint16_t *ranks, uint32_t *counts,
    uint16_t *warp_hist, int valid_count, int shift, int dtype_kind,
    int descending) {
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  if (tid < kRadixSize)
    counts[tid] = 0;
  __syncthreads();

  for (int base = 0; base < valid_count; base += kThreads) {
    for (int i = tid; i < kWarps * kRadixSize; i += kThreads)
      warp_hist[i] = 0;
    __syncthreads();

    const int item = base + tid;
    const bool valid = item < valid_count;
    const unsigned active = __ballot_sync(0xffffffffU, valid);
    uint32_t digit = 0;
    unsigned peers = 0;
    if (valid) {
      digit = (ordered_key(values[item], dtype_kind, descending) >> shift) &
              0xffU;
      peers = __match_any_sync(active, digit);
      const int leader = __ffs(static_cast<int>(peers)) - 1;
      if (lane == leader)
        warp_hist[warp * kRadixSize + digit] =
            static_cast<uint16_t>(__popc(peers));
    }
    __syncthreads();

    if (valid) {
      uint32_t previous_warps = 0;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) {
        if (w < warp)
          previous_warps += warp_hist[w * kRadixSize + digit];
      }
      ranks[item] = static_cast<uint16_t>(
          counts[digit] + previous_warps +
          __popc(peers & lane_mask_lt()));
    }
    __syncthreads();

    if (tid < kRadixSize) {
      uint32_t chunk_count = 0;
#pragma unroll
      for (int w = 0; w < kWarps; ++w)
        chunk_count += warp_hist[w * kRadixSize + tid];
      counts[tid] += chunk_count;
    }
    __syncthreads();
  }
}

__device__ __forceinline__ void exclusive_bin_scan(uint32_t *counts,
                                                    uint32_t *bin_base) {
  if (threadIdx.x == 0) {
    uint32_t running = 0;
#pragma unroll
    for (int bin = 0; bin < kRadixSize; ++bin) {
      bin_base[bin] = running;
      running += counts[bin];
    }
  }
  __syncthreads();
}

} // namespace

// One CTA sorts one complete row.  This path avoids all global histogram,
// lookback and temporary-buffer traffic for the common segmented-sort sizes.
__device__ void SortSmall(GLOBAL const uint16_t *input,
                          GLOBAL uint16_t *output,
                          GLOBAL int64_t *output_indices, int rows, int n,
                          int dtype_kind, int descending,
                          int return_values) {
  const int row = blockIdx.x;
  if (row >= rows)
    return;

  __shared__ uint16_t values_a[kSmallCapacity];
  __shared__ uint16_t values_b[kSmallCapacity];
  __shared__ uint16_t indices_a[kSmallCapacity];
  __shared__ uint16_t indices_b[kSmallCapacity];
  __shared__ uint16_t ranks[kSmallCapacity];
  __shared__ uint32_t counts[kRadixSize];
  __shared__ uint32_t bin_base[kRadixSize];
  __shared__ uint16_t warp_hist[kWarps * kRadixSize];

  const int tid = threadIdx.x;
  const int64_t row_offset = static_cast<int64_t>(row) * n;
  for (int i = tid; i < n; i += blockDim.x) {
    values_a[i] = input[row_offset + i];
    indices_a[i] = static_cast<uint16_t>(i);
  }
  __syncthreads();

  stable_digit_ranks<kSmallCapacity>(values_a, ranks, counts, warp_hist, n,
                                     0, dtype_kind, descending);
  exclusive_bin_scan(counts, bin_base);
  for (int i = tid; i < n; i += blockDim.x) {
    const uint32_t digit = ordered_key(values_a[i], dtype_kind, descending) & 0xffU;
    const uint32_t dst = bin_base[digit] + ranks[i];
    values_b[dst] = values_a[i];
    indices_b[dst] = indices_a[i];
  }
  __syncthreads();

  stable_digit_ranks<kSmallCapacity>(values_b, ranks, counts, warp_hist, n,
                                     8, dtype_kind, descending);
  exclusive_bin_scan(counts, bin_base);
  for (int i = tid; i < n; i += blockDim.x) {
    const uint32_t digit = ordered_key(values_b[i], dtype_kind, descending) >> 8;
    const uint32_t dst = bin_base[digit] + ranks[i];
    values_a[dst] = values_b[i];
    indices_a[dst] = indices_b[i];
  }
  __syncthreads();

  for (int i = tid; i < n; i += blockDim.x) {
    if (return_values)
      output[row_offset + i] = values_a[i];
    output_indices[row_offset + i] = static_cast<int64_t>(indices_a[i]);
  }
}

// The histogram kernel reads every element once and builds both radix passes.
// It also initializes every lookback slot owned by this (row, tile) CTA.
__device__ void SortHistogram(GLOBAL const uint16_t *input,
                              GLOBAL uint32_t *histogram,
                              GLOBAL unsigned long long *status, int rows,
                              int n, int grid_n, int dtype_kind,
                              int descending) {
  const int linear_tile = blockIdx.x;
  const int tile = linear_tile / rows;
  const int row = linear_tile - tile * rows;
  if (row >= rows || tile >= grid_n)
    return;

  __shared__ uint32_t hist[2 * kRadixSize];
  const int tid = threadIdx.x;
  hist[tid] = 0;
  hist[kRadixSize + tid] = 0;

  status[(((static_cast<int64_t>(0) * rows + row) * grid_n + tile) *
          kRadixSize) + tid] = 0;
  status[(((static_cast<int64_t>(1) * rows + row) * grid_n + tile) *
          kRadixSize) + tid] = 0;
  __syncthreads();

  const int tile_start = tile * kLargeTile;
  const int tile_end = tile_start + kLargeTile < n ? tile_start + kLargeTile : n;
  const int64_t row_offset = static_cast<int64_t>(row) * n;
  for (int col = tile_start + tid; col < tile_end; col += blockDim.x) {
    const uint16_t key = ordered_key(input[row_offset + col], dtype_kind,
                                     descending);
    atomicAdd(&hist[key & 0xffU], 1U);
    atomicAdd(&hist[kRadixSize + (key >> 8)], 1U);
  }
  __syncthreads();

  const int64_t histogram_base =
      (static_cast<int64_t>(row) * grid_n + tile) * 2 * kRadixSize;
  histogram[histogram_base + tid] = hist[tid];
  histogram[histogram_base + kRadixSize + tid] =
      hist[kRadixSize + tid];
}

// One CTA reduces the per-tile histograms for a row, then builds both exclusive
// bin scans.  Per-tile storage avoids a memset and contended global atomics.
__device__ void SortPrefix(GLOBAL const uint32_t *histogram,
                           GLOBAL uint32_t *prefix, int rows, int grid_n) {
  const int row = blockIdx.x;
  if (row >= rows)
    return;

  __shared__ uint32_t totals[2 * kRadixSize];
  const int bin = threadIdx.x;
  uint32_t low = 0;
  uint32_t high = 0;
  for (int tile = 0; tile < grid_n; ++tile) {
    const int64_t base =
        (static_cast<int64_t>(row) * grid_n + tile) * 2 * kRadixSize;
    low += histogram[base + bin];
    high += histogram[base + kRadixSize + bin];
  }
  totals[bin] = low;
  totals[kRadixSize + bin] = high;
  __syncthreads();

  if (bin == 0) {
    for (int pass = 0; pass < 2; ++pass) {
      uint32_t running = 0;
      const int64_t base = static_cast<int64_t>(row) * 2 * kRadixSize +
                           pass * kRadixSize;
#pragma unroll
      for (int scan_bin = 0; scan_bin < kRadixSize; ++scan_bin) {
        prefix[base + scan_bin] = running;
        running += totals[pass * kRadixSize + scan_bin];
      }
    }
  }
}

// One CTA owns one row tile and all 256 bins.  Stable local ranks are combined
// with a vectorized decoupled lookback across earlier tiles.
__device__ void SortSweep(
    GLOBAL const uint16_t *input_values,
    GLOBAL const uint32_t *input_indices,
    GLOBAL uint16_t *output_values,
    GLOBAL uint32_t *output_indices32,
    GLOBAL int64_t *output_indices64,
    GLOBAL const uint32_t *prefix,
    GLOBAL unsigned long long *status, int rows, int n, int grid_n,
    int dtype_kind, int descending, int pass, int return_values) {
  const int linear_tile = blockIdx.x;
  const int tile = linear_tile / rows;
  const int row = linear_tile - tile * rows;
  if (row >= rows || tile >= grid_n)
    return;

  __shared__ uint16_t values[kLargeTile];
  __shared__ uint32_t indices[kLargeTile];
  __shared__ uint16_t ranks[kLargeTile];
  __shared__ uint32_t counts[kRadixSize];
  __shared__ uint32_t tile_prefix[kRadixSize];
  __shared__ uint16_t warp_hist[kWarps * kRadixSize];

  const int tid = threadIdx.x;
  const int tile_start = tile * kLargeTile;
  const int remaining = n - tile_start;
  const int valid_count = kLargeTile < remaining ? kLargeTile : remaining;
  const int64_t row_offset = static_cast<int64_t>(row) * n;
  for (int i = tid; i < valid_count; i += blockDim.x) {
    values[i] = input_values[row_offset + tile_start + i];
    indices[i] = pass == 0 ? static_cast<uint32_t>(tile_start + i)
                            : input_indices[row_offset + tile_start + i];
  }
  __syncthreads();

  stable_digit_ranks<kLargeTile>(values, ranks, counts, warp_hist, valid_count,
                                 pass * 8, dtype_kind, descending);

  const uint32_t local_count = counts[tid];
  GLOBAL unsigned long long *current =
      status + ((((static_cast<int64_t>(pass) * rows + row) * grid_n + tile) *
                 kRadixSize) + tid);
  atomicExch(current, kAggregate | static_cast<unsigned long long>(local_count));
  __threadfence();

  unsigned long long exclusive = 0;
  for (int previous = tile - 1; previous >= 0; --previous) {
    GLOBAL unsigned long long *slot =
        status + ((((static_cast<int64_t>(pass) * rows + row) * grid_n +
                    previous) * kRadixSize) + tid);
    unsigned long long packed;
    do {
      packed = atomicCAS(slot, 0ULL, 0ULL);
    } while (packed == 0);
    exclusive += packed & kValueMask;
    if ((packed & kInclusive) != 0)
      break;
  }

  tile_prefix[tid] = static_cast<uint32_t>(exclusive);
  atomicExch(current, kInclusive | (exclusive + local_count));
  __threadfence();
  __syncthreads();

  const int64_t prefix_base = static_cast<int64_t>(row) * 2 * kRadixSize +
                              pass * kRadixSize;
  for (int i = tid; i < valid_count; i += blockDim.x) {
    const uint16_t key = ordered_key(values[i], dtype_kind, descending);
    const uint32_t digit = pass == 0 ? key & 0xffU : key >> 8;
    const uint32_t dst_col = prefix[prefix_base + digit] +
                             tile_prefix[digit] + ranks[i];
    const int64_t dst = row_offset + dst_col;
    if (pass == 0) {
      output_values[dst] = values[i];
      output_indices32[dst] = indices[i];
    } else {
      if (return_values)
        output_values[dst] = values[i];
      output_indices64[dst] = static_cast<int64_t>(indices[i]);
    }
  }
}
