#pragma once

#include "../block_binfly.cuh"
#include "binary_search.cuh"
#include "utils.cuh"
#include <cstdint>

namespace binfly
{

//--------------------------------------------------//
// Block-wide search, for a full tile
//--------------------------------------------------//
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
  auto cached_start       = start;
  auto block              = cg::this_thread_block();
  auto warp               = cg::tiled_partition<warp_threads>(block);

  // If the search space fits in shared memory, do the binary search there
  const index_t num_indices = end - start;
  if (num_indices <= max_smem_keys)
  {
    is_search_data_block_shared = true;

    // Load the key range into shared memory
#pragma unroll
    for (std::int32_t idx = 0; idx < smem_multiplier; ++idx)
    {
      const std::uint32_t smem_idx = block.thread_rank() + idx * BlockThreads;
      if (smem_idx < num_indices)
      {
        storage.search_data.block[smem_idx] = search_data[start + smem_idx];
      }
    }

    // Update the search_data pointer and the start and end indices
    search_data_alias = storage.search_data.block;
    cached_start      = start;
    start             = 0;
    end               = num_indices;
  }

  // Update start for all but the first warp
  search_indices[0] = start;
  if (warp.meta_group_rank() != 0)
  {
    if (warp.thread_rank() == 0)
    {
      search_indices[0] = binary_search(search_data_alias, search_keys[0], start, end);
    }
    start = warp.shfl(search_indices[0], 0);

    // Put start in shared memory to propagate down as end for prior warp
    storage.warp_ends[warp.meta_group_rank() - 1] = start + 1; // + 1 because the end is exclusive
  }
  __syncthreads();

  // Update end for all but the last warp by propagating start down one warp
  if (warp.meta_group_rank() != warp.meta_group_size() - 1)
  {
    end = storage.warp_ends[warp.meta_group_rank()];
  }

  // Invoke warp search
  warp_search(warp, search_indices, search_data_alias, search_keys, start, end);

  // If the search space is block shared, update the search indices
  if (is_search_data_block_shared)
  {
#pragma unroll
    for (std::int32_t idx = 0; idx < ItemsPerThread; ++idx)
    {
      search_indices[idx] += cached_start
    }
  }
}

//--------------------------------------------------//
// Block-wide search, for a partial tile
//--------------------------------------------------//
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
  auto cached_start          = start;
  auto block                 = cg::this_thread_block();
  auto warp                  = cg::tiled_partition<warp_threads>(block);

  // If the search space fits in shared memory, do the binary search there
  const index_t num_indices = end - start;
  if (num_indices <= max_smem_keys)
  {
    is_search_data_block_shared = true;

    // Load the key range into shared memory
#pragma unroll
    for (std::int32_t idx = 0; idx < smem_multiplier; ++idx)
    {
      const std::uint32_t smem_idx = block.thread_rank() + idx * BlockThreads;
      if (smem_idx < num_indices)
      {
        storage.search_data[smem_idx] = search_data[start + smem_idx];
      }
    }

    // Update the search_data pointer and the start and end indices
    search_data_alias = storage.search_data;
    cached_start      = start;
    start             = 0;
    end               = num_indices;
  }

  // Update start for all but the first warp
  search_indices[0] = start;
  if (warp.meta_group_rank() != 0 && warp.meta_group_rank() < num_valid_warps)
  {
    if (warp.thread_rank() == 0)
    {
      search_indices[0] = binary_search(search_data_alias, search_keys[0], start, end);
    }
    start = warp.shfl(search_indices[0], 0);

    // Put start in shared memory to propagate down as end for prior warp
    storage.warp_ends[warp.meta_group_rank() - 1] = start + 1; // + 1 because the end is exclusive
  }
  __syncthreads();

  // Update end for all but the last warp by propagating start down one warp
  if (warp.meta_group_rank() < num_valid_warps - 1)
  {
    //----------Not the Last Warp----------//
    end = storage.warp_ends[warp.meta_group_rank()];
    warp_search(warp, search_indices, search_data_alias, search_keys, start, end);
  }
  else if (warp.meta_group_rank() == num_valid_warps - 1)
  {
    //----------The Last Warp----------//
    // Just do binary search for simplicity
    std::uint32_t local_idx = block.thread_rank() * ItemsPerThread;
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

  // If the search space is block shared, update the search indices to account for the cached start
  if (is_search_data_block_shared)
  {
#pragma unroll
    for (std::int32_t idx = 0; idx < ItemsPerThread; ++idx)
    {
      search_indices[idx] += cached_start;
    }
  }
}

} // namespace binfly