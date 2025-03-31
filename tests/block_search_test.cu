#include <binfly/block_binfly.cuh>
#include <cstdint>
#include <cub/cub.cuh>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

using key_t                             = std::int32_t;
using index_t                           = std::uint32_t;
constexpr std::uint32_t smem_multiplier = 1;

// **Kernel for Block Search**
template <std::uint32_t BlockThreads, std::uint32_t ItemsPerThread>
__global__ void test_block_search_kernel(const key_t* search_data,
                                         const key_t* search_keys,
                                         index_t num_valid_keys,
                                         index_t* search_indices,
                                         index_t start,
                                         index_t end)
{
  static constexpr std::uint32_t block_threads    = BlockThreads;
  static constexpr std::uint32_t items_per_thread = ItemsPerThread;
  static constexpr std::uint32_t tile_items       = block_threads * items_per_thread;

  using block_binfly_t =
    binfly::BlockBinfly<block_threads, items_per_thread, key_t, index_t, smem_multiplier>;
  using block_load_t =
    cub::BlockLoad<key_t, block_threads, items_per_thread, cub::BLOCK_LOAD_DIRECT>;
  using block_load_storage_t = typename block_load_t::TempStorage;
  using block_store_t =
    cub::BlockStore<index_t, block_threads, items_per_thread, cub::BLOCK_STORE_DIRECT>;
  using block_store_storage_t  = typename block_store_t::TempStorage;
  using block_binfly_storage_t = typename block_binfly_t::temp_storage_t;

  __shared__ block_load_storage_t load_storage;
  __shared__ block_store_storage_t store_storage;
  __shared__ block_binfly_storage_t binfly_storage;

  key_t thread_search_keys[items_per_thread];
  index_t thread_search_indices[items_per_thread];

  // Load keys
  block_load_t(load_storage).Load(search_keys, thread_search_keys, num_valid_keys);

  // Do block search
  if (num_valid_keys < tile_items)
  {
    block_binfly_t(binfly_storage)
      .block_search(thread_search_indices,
                    search_data,
                    thread_search_keys,
                    num_valid_keys,
                    start,
                    end);
  }
  else
  {
    // Partial tile
    block_binfly_t(binfly_storage)
      .block_search(thread_search_indices, search_data, thread_search_keys, start, end);
  }

  // Store results
  block_store_t(store_storage).Store(search_indices, thread_search_indices, num_valid_keys);
}

template <std::uint32_t BlockThreads, std::uint32_t ItemsPerThread>
class BlockBinflyTest : public ::testing::Test
{
protected:
  static constexpr std::uint32_t block_threads    = BlockThreads;
  static constexpr std::uint32_t items_per_thread = ItemsPerThread;

  void run_block_search_test(const thrust::host_vector<key_t>& search_data_h,
                             const thrust::host_vector<key_t>& search_keys_h,
                             index_t num_valid_keys,
                             thrust::host_vector<index_t>& search_indices_h,
                             index_t start,
                             index_t end)
  {
    // Device data
    thrust::device_vector<key_t> search_data      = search_data_h;
    thrust::device_vector<key_t> search_keys      = search_keys_h;
    thrust::device_vector<index_t> search_indices = search_indices_h;

    test_block_search_kernel<block_threads, items_per_thread>
      <<<1, block_threads>>>(thrust::raw_pointer_cast(search_data.data()),
                             thrust::raw_pointer_cast(search_keys.data()),
                             num_valid_keys,
                             thrust::raw_pointer_cast(search_indices.data()),
                             start,
                             end);
    CubDebugExit(cudaDeviceSynchronize());

    search_indices_h = search_indices;
  }

  void verify_results(const thrust::host_vector<index_t>& search_indices,
                      const thrust::host_vector<key_t>& search_data,
                      const thrust::host_vector<key_t>& search_keys,
                      index_t num_search_keys)
  {
    for (auto i = 0; i < (std::int32_t)num_search_keys; ++i)
    {
      index_t expected = binfly::binary_search(thrust::raw_pointer_cast(search_data.data()),
                                               search_keys[i],
                                               (index_t)0,
                                               (index_t)search_data.size());
      EXPECT_EQ(search_indices[i], expected) << "Key: " << search_keys[i] << ", Index: " << i;
    }
  }
};

using binfly_test_t = BlockBinflyTest<64, 2>;

// **Test 1: Full tile, search data fits in shared memory**
TEST_F(binfly_test_t, FullTileSmem)
{
  constexpr std::uint32_t block_threads    = binfly_test_t::block_threads;
  constexpr std::uint32_t items_per_thread = binfly_test_t::items_per_thread;
  constexpr std::uint32_t num_search_keys  = block_threads * items_per_thread;

  thrust::host_vector<key_t> search_data = {0, 7, 13, 44, 45, 55, 67, 91};
  thrust::host_vector<key_t> search_keys(num_search_keys);
  thrust::sequence(search_keys.begin(), search_keys.end(), 2, 3);
  thrust::host_vector<index_t> search_indices(num_search_keys);

  run_block_search_test(search_data,
                        search_keys,
                        num_search_keys,
                        search_indices,
                        0,
                        search_data.size());

  verify_results(search_indices, search_data, search_keys, num_search_keys);
}

// **Test 2: Full tile, search data does NOT fit in shared memory**
TEST_F(binfly_test_t, FillTileGmem)
{
  constexpr std::uint32_t block_threads    = binfly_test_t::block_threads;
  constexpr std::uint32_t items_per_thread = binfly_test_t::items_per_thread;
  constexpr std::uint32_t num_search_keys  = block_threads * items_per_thread;

  thrust::host_vector<key_t> search_data(num_search_keys);
  thrust::sequence(search_data.begin(), search_data.end(), 0, 4);
  thrust::host_vector<key_t> search_keys(num_search_keys);
  thrust::sequence(search_keys.begin(), search_keys.end(), 2, 3);
  thrust::host_vector<index_t> search_indices(num_search_keys);

  run_block_search_test(search_data,
                        search_keys,
                        num_search_keys,
                        search_indices,
                        0,
                        search_data.size());

  verify_results(search_indices, search_data, search_keys, num_search_keys);
}

// **Test 3: Partial tile, search data fits in shared memory**
TEST_F(binfly_test_t, PartialTileSmem)
{
  constexpr std::uint32_t block_threads    = binfly_test_t::block_threads;
  constexpr std::uint32_t items_per_thread = binfly_test_t::items_per_thread;
  constexpr std::uint32_t num_search_keys  = block_threads * items_per_thread - 7;

  thrust::host_vector<key_t> search_data = {0, 7, 13, 44, 45, 55, 67, 91};
  thrust::host_vector<key_t> search_keys(num_search_keys);
  thrust::sequence(search_keys.begin(), search_keys.end(), 2, 3);
  thrust::host_vector<index_t> search_indices(num_search_keys);

  run_block_search_test(search_data,
                        search_keys,
                        num_search_keys,
                        search_indices,
                        0,
                        search_data.size());

  verify_results(search_indices, search_data, search_keys, num_search_keys);
}

// **Test 4: Partial tile, search data fits in shared memory**
TEST_F(binfly_test_t, PartialTileGmem)
{
  constexpr std::uint32_t block_threads    = binfly_test_t::block_threads;
  constexpr std::uint32_t items_per_thread = binfly_test_t::items_per_thread;
  constexpr std::uint32_t num_search_keys  = block_threads * items_per_thread - 7;

  thrust::host_vector<key_t> search_data(num_search_keys);
  thrust::sequence(search_data.begin(), search_data.end(), 0, 4);
  thrust::host_vector<key_t> search_keys(num_search_keys);
  thrust::sequence(search_keys.begin(), search_keys.end(), 2, 3);
  thrust::host_vector<index_t> search_indices(num_search_keys);

  run_block_search_test(search_data,
                        search_keys,
                        num_search_keys,
                        search_indices,
                        0,
                        search_data.size());

  verify_results(search_indices, search_data, search_keys, num_search_keys);
}