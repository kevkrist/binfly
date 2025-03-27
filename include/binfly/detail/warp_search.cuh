#pragma once

#include "../block_binfly.cuh"
#include "binary_search.cuh"
#include <cstdint>
#include <cub/cub.cuh>

namespace binfly
{

template <std::uint32_t BlockThreads,
          std::uint32_t ItemsPerThread,
          typename KeyT,
          typename IndexT,
          std::uint32_t SmemMultiplier>
__device__ __forceinline__ void
BlockBinfly<BlockThreads, ItemsPerThread, KeyT, IndexT, SmemMultiplier>::warp_search(
  warp_t warp,
  IndexT (&search_indices)[ItemsPerThread],
  const KeyT* search_data,
  const KeyT (&search_keys)[ItemsPerThread],
  IndexT start,
  IndexT end)
{
  // Binary search the first layer of keys (except the first thread)
  if (warp.thread_rank() != 0)
  {
    search_indices[0] = binary_search(search_data, search_keys[0], start, end);
  }

  // Shift the end indices down
  const IndexT temp = warp.shfl_down(search_indices[0], 1);
  if (warp.thread_rank() != warp.size() - 1)
  {
    end = temp;
  }

  // Propagate the start index from the previous search as you climb the key layers
#pragma unroll
  for (std::int32_t idx = 1; idx < ItemsPerThread; ++idx)
  {
    search_indices[idx] =
      binary_search(search_data, search_keys[idx], search_indices[idx - 1], end);
  }
}

} // namespace binfly