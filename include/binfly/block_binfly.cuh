#pragma once

#include <binfly/detail/utils.cuh>
#include <cooperative_groups.h>
#include <cstdint>
#include <type_traits>

namespace binfly
{

namespace cg = cooperative_groups;

template <std::uint32_t BlockThreads,
          std::uint32_t ItemsPerThread,
          typename KeyT,
          typename IndexT,
          std::uint32_t SmemMultiplier = 4>
class BlockBinfly
{
  static_assert(BlockThreads % warp_threads == 0);
  static_assert(SmemMultiplier > 0);

  using key_t   = KeyT;
  using index_t = IndexT;
  using warp_t  = cg::thread_block_tile<warp_threads, cg::thread_block>;

  static constexpr std::uint32_t block_threads      = BlockThreads;
  static constexpr std::uint32_t items_per_thread   = ItemsPerThread;
  static constexpr std::uint32_t tile_items         = block_threads * items_per_thread;
  static constexpr std::uint32_t smem_multiplier    = SmemMultiplier;
  static constexpr std::uint32_t max_smem_keys      = block_threads * smem_multiplier;
  static constexpr std::uint32_t max_warp_smem_keys = warp_threads * smem_multiplier;
  static constexpr std::uint32_t num_warps          = get_num_warps<block_threads>();

  struct TempStorage
  {
    union
    {
      key_t block[max_smem_keys];
      key_t warp[num_warps][max_warp_smem_keys];
    } search_data;
    index_t warp_ends[cuda::std::max(num_warps - 1,
                                     static_cast<std::uint32_t>(1))]; // For each warp but the last
  };

public:
  using temp_storage_t = TempStorage;

  __device__ BlockBinfly(TempStorage& temp_storage)
      : storage{temp_storage}
  {}

  // For full tile
  __device__ __forceinline__ void block_search(IndexT (&search_indices)[ItemsPerThread],
                                               const KeyT* search_data,
                                               const KeyT (&search_keys)[ItemsPerThread],
                                               IndexT start,
                                               IndexT end);

  // For partial tial
  __device__ __forceinline__ void block_search(IndexT (&search_indices)[ItemsPerThread],
                                               const KeyT* search_data,
                                               const KeyT (&search_keys)[ItemsPerThread],
                                               IndexT num_valid_keys,
                                               IndexT start,
                                               IndexT end);

  __device__ __forceinline__ void warp_search(warp_t group,
                                              IndexT (&search_indices)[ItemsPerThread],
                                              const KeyT* search_data,
                                              const KeyT (&search_keys)[ItemsPerThread],
                                              IndexT start,
                                              IndexT end);

private:
  temp_storage_t& storage;
  bool is_search_data_block_shared = false;
  bool is_search_data_warp_shared  = false;
};

} // namespace binfly

#include "detail/block_search.cuh"
#include "detail/warp_search.cuh"