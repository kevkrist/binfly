#define CUB_STDERR

#include <binfly/device_binfly.cuh>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda/std/__algorithm/min.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

using key_t                            = std::int32_t;
using index_t                          = std::uint32_t;
constexpr std::int32_t smem_multiplier = 1;

// **Kernel for Block Search**
template <std::int32_t BlockThreads, std::int32_t ItemsPerThread>
__global__ void test_device_binfly_kernel(const key_t* search_data,
                                          const key_t* search_keys,
                                          index_t num_search_keys,
                                          index_t* search_indices,
                                          const index_t* tile_starts)
{
  static constexpr auto block_threads    = BlockThreads;
  static constexpr auto items_per_thread = ItemsPerThread;
  static constexpr auto tile_items       = block_threads * items_per_thread;
  using device_binfly_t = binfly::DeviceBinfly<block_threads, items_per_thread, key_t, index_t>;
  using block_binfly_t = typename device_binfly_t::template BlockBinflyPartitioned<smem_multiplier>;
  using block_load_t =
    cub::BlockLoad<key_t, block_threads, items_per_thread, cub::BLOCK_LOAD_DIRECT>;
  using block_store_t =
    cub::BlockStore<index_t, block_threads, items_per_thread, cub::BLOCK_STORE_DIRECT>;
  using block_load_storage_t   = typename block_load_t::TempStorage;
  using block_store_storage_t  = typename block_store_t::TempStorage;
  using block_binfly_storage_t = typename block_binfly_t::temp_storage_t;

  __shared__ block_load_storage_t load_storage;
  __shared__ block_store_storage_t store_storage;
  __shared__ block_binfly_storage_t binfly_storage;

  key_t thread_search_keys[items_per_thread];
  index_t thread_search_indices[items_per_thread];
  index_t num_valid_keys =
    cuda::std::min(static_cast<std::int32_t>(num_search_keys - blockIdx.x * tile_items),
                   tile_items);
  auto offset = blockIdx.x * tile_items;

  // Load keys
  block_load_t(load_storage).Load(search_keys + offset, thread_search_keys, num_valid_keys);

  // Do block search
  if (num_valid_keys < tile_items)
  {
    block_binfly_t(binfly_storage)
      .block_search(thread_search_indices,
                    search_data,
                    thread_search_keys,
                    num_valid_keys,
                    tile_starts);
  }
  else
  {
    // Partial tile
    block_binfly_t(binfly_storage)
      .block_search(thread_search_indices, search_data, thread_search_keys, tile_starts);
  }

  // Store results
  block_store_t(store_storage)
    .Store(search_indices + offset, thread_search_indices, num_valid_keys);
}

template <std::int32_t BlockThreads, std::int32_t ItemsPerThread>
class DeviceBinflyTest : public ::testing::Test
{
protected:
  static constexpr auto block_threads    = BlockThreads;
  static constexpr auto items_per_thread = ItemsPerThread;

  void run_device_search_test(const thrust::host_vector<key_t>& search_data_h,
                              const thrust::host_vector<key_t>& search_keys_h,
                              thrust::host_vector<index_t>& search_indices_h)
  {
    constexpr auto tile_items = block_threads * items_per_thread;

    const auto num_search_keys = static_cast<index_t>(search_keys_h.size());
    const auto num_search_data = static_cast<index_t>(search_data_h.size());
    const auto num_tiles       = cuda::ceil_div(num_search_keys, static_cast<index_t>(tile_items));
    using device_binfly_t = binfly::DeviceBinfly<block_threads, items_per_thread, key_t, index_t>;

    // Device data
    thrust::device_vector<key_t> search_data = search_data_h;
    thrust::device_vector<key_t> search_keys = search_keys_h;
    thrust::device_vector<index_t> search_indices(num_search_keys);

    // Initialize tile starts
    std::size_t num_tile_starts = 0;
    CubDebugExit(device_binfly_t::tile_starts(nullptr,
                                              num_tile_starts,
                                              thrust::raw_pointer_cast(search_keys.data()),
                                              num_search_keys,
                                              thrust::raw_pointer_cast(search_data.data()),
                                              num_search_data));
    thrust::device_vector<index_t> tile_starts(num_tile_starts);
    CubDebugExit(device_binfly_t::tile_starts(thrust::raw_pointer_cast(tile_starts.data()),
                                              num_tile_starts,
                                              thrust::raw_pointer_cast(search_keys.data()),
                                              num_search_keys,
                                              thrust::raw_pointer_cast(search_data.data()),
                                              num_search_data));
    CubDebugExit(cudaDeviceSynchronize());

    test_device_binfly_kernel<block_threads, items_per_thread>
      <<<num_tiles, block_threads>>>(thrust::raw_pointer_cast(search_data.data()),
                                     thrust::raw_pointer_cast(search_keys.data()),
                                     num_search_keys,
                                     thrust::raw_pointer_cast(search_indices.data()),
                                     thrust::raw_pointer_cast(tile_starts.data()));
    CubDebugExit(cudaDeviceSynchronize());

    // Copy results back to host
    search_indices_h = search_indices;
  }

  void verify_results(const thrust::host_vector<index_t>& search_indices,
                      const thrust::host_vector<key_t>& search_data,
                      const thrust::host_vector<key_t>& search_keys)
  {
    const index_t start_index  = 0;
    const auto num_search_data = static_cast<index_t>(search_data.size());

    for (std::size_t i = 0; i < search_keys.size(); ++i)
    {
      index_t expected = binfly::binary_search(thrust::raw_pointer_cast(search_data.data()),
                                               search_keys[i],
                                               start_index,
                                               num_search_data);
      EXPECT_EQ(search_indices[i], expected) << "Key: " << search_keys[i] << ", Index: " << i;
    }
  }
};

using device_binfly_test_t = DeviceBinflyTest<32, 1>;

// **Test 1: Full tiles, search data fits in shared memory**
TEST_F(device_binfly_test_t, FullTileSmem)
{
  constexpr auto block_threads    = device_binfly_test_t::block_threads;
  constexpr auto items_per_thread = device_binfly_test_t::items_per_thread;
  constexpr auto num_search_keys  = 2 * block_threads * items_per_thread; // 2 tiles

  thrust::host_vector<key_t> search_data = {0, 7, 13, 44, 45, 55, 67, 91};
  thrust::host_vector<key_t> search_keys(num_search_keys);
  thrust::sequence(search_keys.begin(), search_keys.end(), 2, 3);
  thrust::host_vector<index_t> search_indices(num_search_keys);

  run_device_search_test(search_data, search_keys, search_indices);

  verify_results(search_indices, search_data, search_keys);
}

// **Test 2: Full tile, search data does NOT fit in shared memory**
TEST_F(device_binfly_test_t, FillTileGmem)
{
  constexpr auto block_threads    = device_binfly_test_t::block_threads;
  constexpr auto items_per_thread = device_binfly_test_t::items_per_thread;
  constexpr auto num_search_keys  = 2 * block_threads * items_per_thread;

  thrust::host_vector<key_t> search_data(num_search_keys);
  thrust::sequence(search_data.begin(), search_data.end(), 0, 4);
  thrust::host_vector<key_t> search_keys(num_search_keys);
  thrust::sequence(search_keys.begin(), search_keys.end(), 2, 3);
  thrust::host_vector<index_t> search_indices(num_search_keys);

  run_device_search_test(search_data, search_keys, search_indices);

  verify_results(search_indices, search_data, search_keys);
}

// **Test 3: Partial tile, search data fits in shared memory**
TEST_F(device_binfly_test_t, PartialTileSmem)
{
  constexpr auto block_threads    = device_binfly_test_t::block_threads;
  constexpr auto items_per_thread = device_binfly_test_t::items_per_thread;
  constexpr auto num_search_keys  = 2 * block_threads * items_per_thread - 7;

  thrust::host_vector<key_t> search_data = {0, 7, 13, 44, 45, 55, 67, 91};
  thrust::host_vector<key_t> search_keys(num_search_keys);
  thrust::sequence(search_keys.begin(), search_keys.end(), 2, 3);
  thrust::host_vector<index_t> search_indices(num_search_keys);

  run_device_search_test(search_data, search_keys, search_indices);

  verify_results(search_indices, search_data, search_keys);
}

// **Test 4: Partial tile, search data does NOT fit in shared memory**
TEST_F(device_binfly_test_t, PartialTileGmem)
{
  constexpr auto block_threads    = device_binfly_test_t::block_threads;
  constexpr auto items_per_thread = device_binfly_test_t::items_per_thread;
  constexpr auto num_search_keys  = 2 * block_threads * items_per_thread - 7;

  thrust::host_vector<key_t> search_data(num_search_keys);
  thrust::sequence(search_data.begin(), search_data.end(), 0, 4);
  thrust::host_vector<key_t> search_keys(num_search_keys);
  thrust::sequence(search_keys.begin(), search_keys.end(), 2, 3);
  thrust::host_vector<index_t> search_indices(num_search_keys);

  run_device_search_test(search_data, search_keys, search_indices);

  verify_results(search_indices, search_data, search_keys);
}