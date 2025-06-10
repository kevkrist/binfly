#include <binfly/detail/binary_search.cuh>
#include <cstdint>
#include <cub/cub.cuh>
#include <gtest/gtest.h>
#include <thrust/host_vector.h>

namespace binfly
{
using index_t                                = std::int32_t;
using key_t                                  = std::int32_t;

// **Test Fixture for Binary Search**
class BinarySearchTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    sorted_data = {1, 3, 5, 7, 9, 11, 13, 17};
    end         = index_t(sorted_data.size());
  }

  thrust::host_vector<key_t> sorted_data;
  index_t end;
};

// **Test Case 1: Search for an existing element**
TEST_F(BinarySearchTest, FindsExactMatch)
{
  key_t key              = 7;
  index_t expected_index = 3; // sorted_data[3] == 7

  index_t found_index = binary_search(sorted_data.data(), key, index_t(0), end);

  EXPECT_EQ(found_index, expected_index);
}

// **Test Case 2: Search for a non-existing element**
TEST_F(BinarySearchTest, FindsInsertionPoint)
{
  key_t key              = 8;
  index_t expected_index = 3; // sorted_data[3] <= 8 < sorted_data[4]

  index_t found_index = binary_search(sorted_data.data(), key, index_t(0), end);

  EXPECT_EQ(found_index, expected_index);
}

// **Test Case 3: Search for the smallest element**
TEST_F(BinarySearchTest, FindsFirstElement)
{
  key_t key              = 1;
  index_t expected_index = 0;

  index_t found_index = binary_search(sorted_data.data(), key, index_t(0), end);

  EXPECT_EQ(found_index, expected_index);
}

// **Test Case 4: Search for the largest element**
TEST_F(BinarySearchTest, FindsLastElement)
{
  key_t key              = 17;
  index_t expected_index = 7;

  index_t found_index = binary_search(sorted_data.data(), key, index_t(0), end);

  EXPECT_EQ(found_index, expected_index);
}

// **Test Case 5: Search for an out-of-bounds element (larger)**
TEST_F(BinarySearchTest, FindsEndForLargerElement)
{
  key_t key              = 20;
  index_t expected_index = end - 1; // Should be placed at last index

  index_t found_index = binary_search(sorted_data.data(), key, index_t(0), end);

  EXPECT_EQ(found_index, expected_index);
}

// **Test Case 6: Search for an out-of-bounds element (smaller)**
TEST_F(BinarySearchTest, FindsStartForSmallerElement)
{
  key_t key              = -5;
  index_t expected_index = 0; // Should be placed at the first index

  index_t found_index = binary_search(sorted_data.data(), key, index_t(0), end);

  EXPECT_EQ(found_index, expected_index);
}

} // namespace binfly

// **Main function for Google Test**
std::int32_t main(std::int32_t argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
