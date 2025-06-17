#define CUB_STDERR

#include <algorithm>
#include <binfly/detail/binary_search.cuh>
#include <binfly/detail/partitioned_search.cuh>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

// **Test 1**
TEST(PartitionedSearchTest, ThreeTileSearch)
{
  /// CONFIG ///
  using key_t   = std::uint32_t;
  using index_t = std::int32_t;

  constexpr std::int32_t block_threads    = 4;
  constexpr std::int32_t items_per_thread = 1;
  constexpr int32_t num_tiles             = 3;
  /// END CONFIG ///

  constexpr auto tile_items                = block_threads * items_per_thread;
  constexpr auto num_search_keys           = num_tiles * tile_items - 1;
  constexpr auto num_ctas                  = num_tiles + 1;
  thrust::device_vector<key_t> search_data = {5, 10, 15};
  thrust::device_vector<key_t> search_keys(num_search_keys);
  thrust::sequence(search_keys.begin(), search_keys.end(), 0, 2);
  thrust::device_vector<index_t> tile_starts(num_ctas);

  binfly::partitioned_search<tile_items>
    <<<num_ctas, block_threads>>>(thrust::raw_pointer_cast(tile_starts.data()),
                                  num_tiles,
                                  thrust::raw_pointer_cast(search_keys.data()),
                                  static_cast<index_t>(search_keys.size()),
                                  thrust::raw_pointer_cast(search_data.data()),
                                  static_cast<index_t>(search_data.size()));
  CubDebugExit(cudaDeviceSynchronize());

  // Validate result on host
  thrust::host_vector<index_t> tile_starts_h = tile_starts;
  thrust::host_vector<key_t> search_data_h   = search_data;
  thrust::host_vector<key_t> search_keys_h   = search_keys;
  for (auto tile = 0; tile <= num_ctas; ++tile)
  {
    const index_t key_idx  = std::min(tile * tile_items, num_search_keys - 1);
    const index_t expected = binfly::binary_search(thrust::raw_pointer_cast(search_data_h.data()),
                                                   search_keys_h[key_idx],
                                                   0,
                                                   static_cast<index_t>(search_data_h.size()));
    EXPECT_EQ(tile_starts_h[tile], expected) << "Tile " << tile;
  }
}