#pragma once

#include <binfly/block_binfly.cuh>
#include <binfly/detail/binary_search.cuh>
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
  auto* search_data_alias = const_cast<key_t*>(search_data);
  auto cached_start       = start;

  // If the search space for the block doesn't fit in shared memory, check if the search space for
  // the warp fits in shared memory
  if (!is_search_data_block_shared)
  {
    is_search_data_warp_shared = true;

    const index_t num_indices = end - start;
    if (num_indices <= max_warp_smem_keys)
    {
// Load the key range into shared memory
#pragma unroll
      for (std::int32_t idx = 0; idx < smem_multiplier; ++idx)
      {
        const std::uint32_t smem_idx = warp.thread_rank() + idx * warp_threads;
        if (smem_idx < num_indices)
        {
          storage.search_data.warp[warp.meta_group_rank()][smem_idx] =
            search_data[start + smem_idx];
        }
      }

      // Update the search_data pointer and the start and end indices
      search_data_alias = storage.search_data.warp[warp.meta_group_rank()];
      cached_start      = start;
      start             = 0;
      end               = num_indices;
    }
  }

  // Binary search the first layer of keys (except the first thread)
  search_indices[0] = start;
  if (warp.thread_rank() != 0)
  {
    search_indices[0] = binary_search(search_data, search_keys[0], start, end);
  }

  // Shift the end indices down
  const IndexT temp = warp.shfl_down(search_indices[0], 1);
  if (warp.thread_rank() != warp.size() - 1)
  {
    end = temp + 1; // + 1 because the end is exclusive
  }

  // Propagate the start index from the previous search as you climb the key layers
#pragma unroll
  for (std::int32_t idx = 1; idx < ItemsPerThread; ++idx)
  {
    search_indices[idx] =
      binary_search(search_data, search_keys[idx], search_indices[idx - 1], end);
  }

  // If the search space is warp_shared, update the search indices to account for the cached start
  if (is_search_data_warp_shared)
  {
#pragma unroll
    for (std::int32_t idx = 0; idx < ItemsPerThread; ++idx)
    {
      search_indices[idx] += cached_start;
    }
  }
}

} // namespace binfly