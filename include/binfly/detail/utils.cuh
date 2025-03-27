#pragma once

#include "cub/util_math.cuh"
#include <cstdint>
#include <cub/cub.cuh>

namespace binfly
{

static constexpr std::uint32_t warp_threads     = 32;
static constexpr std::uint32_t log_warp_threads = 5;
static constexpr std::uint32_t last_warp_thread = 31;

template <std::uint32_t BlockThreads>
__host__ __device__ constexpr std::uint32_t get_num_warps()
{
  return BlockThreads >> log_warp_threads;
}

template <std::uint32_t ItemsPerThread>
__host__ __device__ std::uint32_t get_num_warps(std::uint32_t num_valid_items)
{
  const auto num_valid_threads = cub::DivideAndRoundUp(num_valid_items, ItemsPerThread);
  return cub::DivideAndRoundUp(num_valid_threads, warp_threads);
}

} // namespace binfly