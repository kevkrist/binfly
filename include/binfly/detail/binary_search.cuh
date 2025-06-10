#pragma once

#include <cub/cub.cuh>
#include <cuda/std/type_traits>

namespace binfly
{

// Binary search in the range [start, end)
template <typename T, typename IndexT>
__host__ __device__ __forceinline__ IndexT
binary_search(const T* search_data, const T& search_key, IndexT start, IndexT end)
{
  // Sentinel key indicating that search_data does not contain a key satisfying <= semantics with
  // respect to search_key.
  static constexpr IndexT sentinel = std::numeric_limits<IndexT>::max();

  IndexT idx;
  T current_key;

  while (start < end)
  {
    idx         = cub::MidPoint(start, end);
    current_key = search_data[idx];

    if (current_key == search_key)
    {
      return idx;
    }
    else if (current_key < search_key)
    {
      start = idx + 1;
    }
    else
    {
      end = idx;
    }
  }

  // start has incremented one beyond the index satisfying <= semantics
  return start == 0 ? sentinel : start - 1;
}

// The hint index is assumed to be in the range [start, end)
template <typename T, typename IndexT>
__host__ __device__ __forceinline__ IndexT
binary_search_hint(const T* search_data, const T& search_key, IndexT hint, IndexT start, IndexT end)
{
  const T hint_key = search_data[hint];

  if (hint_key == search_key)
  {
    return hint;
  }
  else if (hint_key < search_key)
  {
    start = hint + 1;
  }
  else
  {
    end = hint;
  }

  return binary_search(search_data, search_key, start, end);
}

} // namespace binfly
