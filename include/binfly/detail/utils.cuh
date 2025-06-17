#pragma once

#include <cstdint>
#include <cub/cub.cuh>

namespace binfly
{

static constexpr std::int32_t warp_threads = 32;

template <std::int32_t BlockThreads>
__host__ __device__ constexpr std::int32_t get_num_warps()
{
  return BlockThreads / warp_threads;
}

template <std::int32_t ItemsPerThread>
__host__ __device__ std::int32_t get_num_warps(std::int32_t num_valid_items)
{
  const auto num_valid_threads = cuda::ceil_div(num_valid_items, ItemsPerThread);
  return cuda::ceil_div(num_valid_threads, warp_threads);
}

template <typename T, std::int32_t ItemsPerThread>
__device__ __forceinline__ void fill_registers(T (&items)[ItemsPerThread], T value)
{
#pragma unroll
  for (std::int32_t idx = 0; idx < ItemsPerThread; ++idx)
  {
    items[idx] = value;
  }
}

} // namespace binfly