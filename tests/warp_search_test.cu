#include <binfly/block_binfly.cuh>
#include <cooperative_groups.h>
#include <cstdint>
#include <cub/cub.cuh>
#include <gtest/gtest.h>
#include <limits>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

using key_t                              = std::int32_t;
using index_t                            = std::uint32_t;
constexpr std::uint32_t block_threads    = 32; // warp size
constexpr std::uint32_t items_per_thread = 2;
constexpr std::uint32_t num_search_keys  = block_threads * items_per_thread;
constexpr std::uint32_t smem_multiplier  = 1;

using block_binfly_t =
  binfly::BlockBinfly<block_threads, items_per_thread, key_t, index_t, smem_multiplier>;

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
  using block_load_storage_t = typename block_load_t::TempStorage;
  using block_store_t =
    cub::BlockStore<index_t, block_threads, items_per_thread, cub::BLOCK_STORE_DIRECT>;
  using block_store_storage_t  = typename block_store_t::TempStorage;
  using block_binfly_storage_t = typename block_binfly_t::temp_storage_t;

  __shared__ block_load_storage_t load_storage;
  __shared__ block_store_storage_t store_storage;
  __shared__ block_binfly_storage_t binfly_storage;

  key_t thread_keys[items_per_thread];
  index_t thread_indices[items_per_thread];

  // Load keys
  block_load_t(load_storage).Load(search_keys, thread_keys);

  // Do warp search
  auto warp = cg::tiled_partition<block_threads>(cg::this_thread_block());
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
};

// **Test 1**
TEST_F(WarpSearchTest, SortedInputFindsLowerBounds)
{
  const thrust::host_vector<key_t> search_data = {1, 5, 21, 23, 25, 33, 37, 99, 150};
  thrust::host_vector<key_t> search_keys(num_search_keys);
  thrust::sequence(search_keys.begin(), search_keys.end(), 0, 2);
  thrust::host_vector<index_t> search_indices(num_search_keys, std::numeric_limits<index_t>::max());

  run_warp_search_test(search_data, search_keys, search_indices, 0, search_data.size());

  for (auto i = 0; i < (std::int32_t)num_search_keys; ++i)
  {
    auto it          = std::lower_bound(search_data.begin(), search_data.end(), search_keys[i]);
    index_t expected = static_cast<index_t>(std::distance(search_data.begin(), it));
    EXPECT_EQ(search_indices[i], expected) << "Key: " << search_keys[i];
  }
}
