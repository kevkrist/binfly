#include <binfly/detail/binary_search.cuh>
#include <cub/cub.cuh>
#include <gtest/gtest.h>

namespace binfly
{

// **Test Fixture for Binary Search**
class BinarySearchTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    sorted_data = {1, 3, 5, 7, 9, 11, 13, 17};
    size        = sorted_data.size();
  }

  std::vector<std::int32_t> sorted_data;
  std::size_t size;
};

// **Test Case 1: Search for an existing element**
TEST_F(BinarySearchTest, FindsExactMatch)
{
  std::int32_t key            = 7;
  std::int32_t expected_index = 3; // sorted_data[3] == 7

  std::int32_t found_index = binary_search(sorted_data.data(), key, size_t(0), size);

  EXPECT_EQ(found_index, expected_index);
}

// **Test Case 2: Search for a non-existing element**
TEST_F(BinarySearchTest, FindsInsertionPoint)
{
  std::int32_t key            = 8;
  std::int32_t expected_index = 4; // sorted_data[3] <= 8 < sorted_data[4]

  std::int32_t found_index = binary_search(sorted_data.data(), key, size_t(0), size);

  EXPECT_EQ(found_index, expected_index);
}

// **Test Case 3: Search for the smallest element**
TEST_F(BinarySearchTest, FindsFirstElement)
{
  std::int32_t key            = 1;
  std::int32_t expected_index = 0;

  std::int32_t found_index = binary_search(sorted_data.data(), key, size_t(0), size);

  EXPECT_EQ(found_index, expected_index);
}

// **Test Case 4: Search for the largest element**
TEST_F(BinarySearchTest, FindsLastElement)
{
  std::int32_t key            = 17;
  std::int32_t expected_index = 7;

  std::int32_t found_index = binary_search(sorted_data.data(), key, size_t(0), size);

  EXPECT_EQ(found_index, expected_index);
}

// **Test Case 5: Search for an out-of-bounds element (larger)**
TEST_F(BinarySearchTest, FindsEndForLargerElement)
{
  std::int32_t key            = 20;
  std::int32_t expected_index = size; // Should be placed after last element

  std::int32_t found_index = binary_search(sorted_data.data(), key, size_t(0), size);

  EXPECT_EQ(found_index, expected_index);
}

// **Test Case 6: Search for an out-of-bounds element (smaller)**
TEST_F(BinarySearchTest, FindsStartForSmallerElement)
{
  std::int32_t key            = -5;
  std::int32_t expected_index = 0; // Should be placed at the beginning

  std::int32_t found_index = binary_search(sorted_data.data(), key, size_t(0), size);

  EXPECT_EQ(found_index, expected_index);
}

} // namespace binfly

// **Main function for Google Test**
std::int32_t main(std::int32_t argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
