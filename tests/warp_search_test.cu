#define CUB_STDERR

#include <binfly/block_binfly.cuh>
#include <cooperative_groups.h>
#include <cstdint>
#include <cub/cub.cuh>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

using key_t                             = std::int32_t;
using index_t                           = std::int32_t;
constexpr std::int32_t block_threads    = 32; // warp size
constexpr std::int32_t items_per_thread = 2;
constexpr std::int32_t num_search_keys  = block_threads * items_per_thread;
constexpr std::int32_t smem_multiplier  = 1;

// **Kernel for Warp Search**
__global__ void test_warp_search_kernel(const key_t* search_data,
                                        const key_t* search_keys,
                                        index_t* search_indices,
                                        index_t start,
                                        index_t end)
{
  namespace cg = cooperative_groups;
  using block_load_t =
    cub::BlockLoad<key_t, block_threads, items_per_thread, cub::BLOCK_LOAD_DIRECT>;
  using block_store_t =
    cub::BlockStore<index_t, block_threads, items_per_thread, cub::BLOCK_STORE_DIRECT>;
  using block_binfly_t =
    binfly::BlockBinfly<block_threads, items_per_thread, key_t, index_t, smem_multiplier>;
  using block_load_storage_t   = typename block_load_t::TempStorage;
  using block_store_storage_t  = typename block_store_t::TempStorage;
  using block_binfly_storage_t = typename block_binfly_t::temp_storage_t;

  __shared__ block_load_storage_t load_storage;
  __shared__ block_store_storage_t store_storage;
  __shared__ block_binfly_storage_t binfly_storage;
  auto warp = cg::tiled_partition<block_threads>(cg::this_thread_block());

  key_t thread_keys[items_per_thread];
  index_t thread_indices[items_per_thread];

  // Load keys
  block_load_t(load_storage).Load(search_keys, thread_keys);

  // Do warp search
  block_binfly_t(binfly_storage)
    .warp_search(warp, thread_indices, search_data, thread_keys, start, end);

  // Store results
  block_store_t(store_storage).Store(search_indices, thread_indices);
}

// **Test Fixture for Binary Search**
class WarpSearchTest : public ::testing::Test
{
protected:
  void run_warp_search_test(const thrust::host_vector<key_t>& search_data_h,
                            const thrust::host_vector<key_t>& search_keys_h,
                            thrust::host_vector<index_t>& search_indices_h,
                            index_t start,
                            index_t end)
  {
    // Device data
    thrust::device_vector<key_t> search_data_d = search_data_h;
    thrust::device_vector<key_t> search_keys_d = search_keys_h;
    thrust::device_vector<index_t> search_indices_d(num_search_keys);

    test_warp_search_kernel<<<1, block_threads>>>(thrust::raw_pointer_cast(search_data_d.data()),
                                                  thrust::raw_pointer_cast(search_keys_d.data()),
                                                  thrust::raw_pointer_cast(search_indices_d.data()),
                                                  start,
                                                  end);
    CubDebugExit(cudaDeviceSynchronize());

    search_indices_h = search_indices_d;
  }

  void verify_results(const thrust::host_vector<index_t>& search_indices,
                      const thrust::host_vector<key_t>& search_data,
                      const thrust::host_vector<key_t>& search_keys)
  {
    for (auto i = 0; i < (std::int32_t)num_search_keys; ++i)
    {
      index_t expected =
        binfly::binary_search<key_t, index_t>(thrust::raw_pointer_cast(search_data.data()),
                                              search_keys[i],
                                              0,
                                              search_data.size());
      EXPECT_EQ(search_indices[i], expected) << "Key: " << search_keys[i] << ", Index: " << i;
    }
  }
};

// **Test 1: search data not in shared memory**
TEST_F(WarpSearchTest, NoSmem)
{
  thrust::host_vector<key_t> search_data = {1,  5,  6,  7,   8,   11,  13,  17,  18,  21, 23, 25,
                                            33, 37, 50, 51,  52,  53,  54,  55,  56,  57, 58, 59,
                                            60, 61, 99, 101, 104, 107, 111, 123, 133, 150};
  assert(search_data.size() >= smem_multiplier * block_threads);
  thrust::host_vector<key_t> search_keys(num_search_keys);
  thrust::sequence(search_keys.begin(), search_keys.end(), 0, 2);
  thrust::host_vector<index_t> search_indices(num_search_keys);

  run_warp_search_test(search_data, search_keys, search_indices, 0, search_data.size());

  verify_results(search_indices, search_data, search_keys);
}

// **Test 2: search data in shared memory**
TEST_F(WarpSearchTest, YesSmem)
{
  thrust::host_vector<key_t> search_data = {1, 5, 21, 23, 25, 33, 37, 99, 150};
  thrust::host_vector<key_t> search_keys(num_search_keys);
  thrust::sequence(search_keys.begin(), search_keys.end(), 0, 2);
  thrust::host_vector<index_t> search_indices(num_search_keys);

  run_warp_search_test(search_data, search_keys, search_indices, 0, search_data.size());

  verify_results(search_indices, search_data, search_keys);
}
