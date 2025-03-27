#include <algorithm>
#include <binfly/detail/partitioned_search.cuh>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

using key_t   = std::int32_t;
using index_t = std::uint32_t;

index_t host_binary_search(const thrust::host_vector<key_t>& data, key_t key)
{
  return std::lower_bound(data.begin(), data.end(), key) - data.begin();
}

// **Test 1**
TEST(PartitionedSearchTest, ThreeTileSearch)
{
  constexpr std::uint32_t block_threads    = 4;
  constexpr std::uint32_t items_per_thread = 1;
  constexpr std::uint32_t tile_items       = block_threads * items_per_thread;
  constexpr std::uint32_t num_search_keys  = 10;
  constexpr std::uint32_t num_tiles        = cub::DivideAndRoundUp(num_search_keys, tile_items);
  constexpr std::uint32_t num_ctas         = num_tiles + 1;
  thrust::device_vector<key_t> search_data = {5, 10, 15};
  thrust::device_vector<key_t> search_keys(num_search_keys);
  thrust::sequence(search_keys.begin(), search_keys.end(), 0, 2);
  thrust::device_vector<index_t> tile_starts(num_ctas);

  binfly::partitioned_search<tile_items>
    <<<num_ctas, block_threads>>>(thrust::raw_pointer_cast(tile_starts.data()),
                                  num_tiles,
                                  thrust::raw_pointer_cast(search_keys.data()),
                                  (std::uint32_t)search_keys.size(),
                                  thrust::raw_pointer_cast(search_data.data()),
                                  (std::uint32_t)search_data.size());
  CubDebugExit(cudaDeviceSynchronize());

  // Validate result on host
  thrust::host_vector<index_t> tile_starts_h = tile_starts;
  for (index_t tile = 0; tile < num_ctas; ++tile)
  {
    const index_t key_idx  = std::min(tile * tile_items, num_search_keys - 1);
    const index_t expected = host_binary_search(search_data, search_keys[key_idx]);
    EXPECT_EQ(tile_starts_h[tile], expected) << "Tile " << tile;
  }
}
