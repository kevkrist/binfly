#pragma once

#include <binfly/block_binfly.cuh>
#include <binfly/detail/partitioned_search.cuh>
#include <cstdint>
#include <cub/cub.cuh>

namespace binfly
{

template <std::int32_t BlockThreads, std::int32_t ItemsPerThread, typename KeyT, typename IndexT>
struct DeviceBinfly
{
  static constexpr auto block_threads    = BlockThreads;
  static constexpr auto items_per_thread = ItemsPerThread;
  static constexpr auto tile_items       = block_threads * items_per_thread;

  static __host__ cudaError_t tile_starts(IndexT* tile_starts,
                                          std::size_t& num_tile_starts,
                                          const KeyT* search_keys,
                                          IndexT num_search_keys,
                                          const KeyT* search_data,
                                          IndexT num_search_data,
                                          cudaStream_t stream = cudaStreamDefault)
  {
    // Determine the number of tiles of search keys
    const IndexT num_tiles = cuda::ceil_div(num_search_keys, static_cast<IndexT>(tile_items));

    // Determine the allocation requirements
    if (tile_starts == nullptr)
    {
      num_tile_starts = num_tiles + 1;
      return cudaSuccess;
    }

    // Assume correct allocation
    assert(num_tile_starts == num_tiles + 1);
    const std::size_t num_partitioned_search_tiles =
      cuda::ceil_div(num_tile_starts, static_cast<std::size_t>(block_threads));
    partitioned_search<tile_items>
      <<<num_partitioned_search_tiles, block_threads, 0, stream>>>(tile_starts,
                                                                   num_tiles,
                                                                   search_keys,
                                                                   num_search_keys,
                                                                   search_data,
                                                                   num_search_data);

    // Check for execution error
    return CubDebug(cudaGetLastError());
  }

  template <std::uint32_t SmemMultiplier = 4>
  class BlockBinflyPartitioned
  {
    using block_binfly_t =
      BlockBinfly<block_threads, items_per_thread, KeyT, IndexT, SmemMultiplier>;

  public:
    using temp_storage_t = typename block_binfly_t::temp_storage_t;

    __device__ __forceinline__ BlockBinflyPartitioned(temp_storage_t& temp_storage)
        : storage(temp_storage)
    {}

    __device__ __forceinline__ void block_search(IndexT (&search_indices)[ItemsPerThread],
                                                 const KeyT* search_data,
                                                 const KeyT (&search_keys)[ItemsPerThread],
                                                 const IndexT* tile_starts)
    {
      const auto start = tile_starts[blockIdx.x];
      const auto end   = tile_starts[blockIdx.x + 1] + 1; // +1 for exclusive end

      block_binfly_t(storage).block_search(search_indices, search_data, search_keys, start, end);
    }

    // For partial tile
    __device__ __forceinline__ void block_search(IndexT (&search_indices)[ItemsPerThread],
                                                 const KeyT* search_data,
                                                 const KeyT (&search_keys)[ItemsPerThread],
                                                 IndexT num_valid_keys,
                                                 const IndexT* tile_starts)
    {
      const auto start = tile_starts[blockIdx.x];
      const auto end   = tile_starts[blockIdx.x + 1] + 1; // +1 for exclusive end

      block_binfly_t(storage)
        .block_search(search_indices, search_data, search_keys, num_valid_keys, start, end);
    }

  private:
    temp_storage_t& storage;
  };
};

} // namespace binfly
