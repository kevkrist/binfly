#pragma once

#include "../block_binfly.cuh"
#include "binary_search.cuh"
#include "utils.cuh"
#include <cstdint>

namespace binfly
{

template <std::uint32_t BlockThreads,
          std::uint32_t ItemsPerThread,
          typename KeyT,
          typename IndexT,
          std::uint32_t SmemMultiplier>
__device__ __forceinline__ void
BlockBinfly<BlockThreads, ItemsPerThread, KeyT, IndexT, SmemMultiplier>::block_search(
  IndexT (&search_indices)[ItemsPerThread],
  const KeyT* search_data,
  const KeyT (&search_keys)[ItemsPerThread],
  IndexT start,
  IndexT end)
{
  auto* search_data_alias = const_cast<key_t*>(search_data);
  auto warp               = cg::tiled_partition<warp_threads>(cg::this_thread_block());

  // If the search space fits in shared memory, do the binary search there
  const index_t num_indices = end - start;
  if (num_indices <= max_smem_keys)
  {
    // Load the key range into shared memory
#pragma unroll
    for (std::int32_t idx = 0; idx < smem_multiplier; ++idx)
    {
      const std::uint32_t smem_idx = threadIdx.x + idx * BlockThreads;
      if (smem_idx < num_indices)
      {
        storage.search_data[smem_idx] = search_data[start + smem_idx];
      }
    }

    // Update the search_data pointer
    search_data_alias = storage.search_data;
  }

  // Update start for all but the first warp
  if (warp.meta_group_rank() != 0)
  {
    if (warp.thread_rank() == 0)
    {
      search_indices[0] = binary_search(search_data, search_keys[0], start, end);
    }
    start = warp.shfl(search_indices[0], 0);

    // Put start in shared memory to propagate down as end for prior warp
    storage.warp_ends[warp.meta_group_rank() - 1] = start;
  }
  __syncthreads();

  // Update end for all but the last warp by propagating start down one warp
  if (warp.meta_group_rank() != warp.meta_group_size() - 1)
  {
    end = storage.warp_ends[warp.meta_group_rank()];
  }

  // Invoke warp search
  warp_search(warp, search_indices, search_data_alias, search_keys, start, end);
}

template <std::uint32_t BlockThreads,
          std::uint32_t ItemsPerThread,
          typename KeyT,
          typename IndexT,
          std::uint32_t SmemMultiplier>
__device__ __forceinline__ void
BlockBinfly<BlockThreads, ItemsPerThread, KeyT, IndexT, SmemMultiplier>::block_search(
  IndexT (&search_indices)[ItemsPerThread],
  const KeyT* search_data,
  const KeyT (&search_keys)[ItemsPerThread],
  IndexT num_valid_keys,
  IndexT start,
  IndexT end)
{
  const auto num_valid_warps = get_num_warps<ItemsPerThread>(num_valid_keys);
  auto* search_data_alias    = const_cast<key_t*>(search_data);
  auto warp                  = cg::tiled_partition<warp_threads>(cg::this_thread_block());

  // If the search space fits in shared memory, do the binary search there
  const index_t num_indices = end - start;
  if (num_indices <= max_smem_keys)
  {
    // Load the key range into shared memory
#pragma unroll
    for (std::int32_t idx = 0; idx < smem_multiplier; ++idx)
    {
      const std::uint32_t smem_idx = threadIdx.x + idx * BlockThreads;
      if (smem_idx < num_indices)
      {
        storage.search_data[smem_idx] = search_data[start + smem_idx];
      }
    }

    // Update the search_data pointer
    search_data_alias = storage.search_data;
  }

  // Update start for all but the first warp
  if (warp.meta_group_rank() != 0 && warp.meta_group_rank() < num_valid_warps)
  {
    if (warp.thread_rank() == 0)
    {
      search_indices[0] = binary_search(search_data, search_keys[0], start, end);
    }
    start = warp.shfl(search_indices[0], 0);

    // Put start in shared memory to propagate down as end for prior warp
    storage.warp_ends[warp.meta_group_rank() - 1] = start;
  }
  __syncthreads();

  // Update end for all but the last warp by propagating start down one warp
  if (warp.meta_group_rank() < num_valid_warps - 1)
  {
    end = storage.warp_ends[warp.meta_group_rank()];
    warp_search(warp, search_indices, search_data_alias, search_keys, start, end);
  }
  else if (warp.meta_group_rank() == num_valid_warps - 1)
  {
    // Simple binary search for the last warp
    std::uint32_t local_idx = threadIdx.x * ItemsPerThread;
    if (warp.thread_rank() != 0 && local_idx < num_valid_keys)
    {
      search_indices[0] = binary_search(search_data, search_keys[0], start, end);
    }
    ++local_idx;

#pragma unroll
    for (std::int32_t idx = 1; idx < ItemsPerThread; ++idx, ++local_idx)
    {
      if (local_idx < num_valid_keys)
      {
        // KEVIN: consider narrowing the box by propagating starts from below
        search_indices[idx] = binary_search(search_data, search_keys[idx], start, end);
      }
    }
  }
}

} // namespace binfly