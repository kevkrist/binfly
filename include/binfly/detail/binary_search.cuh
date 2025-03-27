#pragma once

#include <cub/cub.cuh>

namespace binfly
{

template <typename T, typename IndexT>
__host__ __device__ __forceinline__ IndexT
binary_search(const T* search_data, const T& search_key, IndexT start, IndexT end)
{
  IndexT idx;
  T current_key;

  while (start < end)
  {
    idx         = cub::MidPoint(start, end);
    current_key = search_data[idx];
    if (search_key <= current_key)
    {
      end = idx;
    }
    else
    {
      start = idx + 1;
    }
  }

  return start;
}

// The hint index is assumed to be in the range [start, end]
template <typename T, typename IndexT>
__host__ __device__ __forceinline__ IndexT
binary_search_hint(const T* search_data, const T& search_key, IndexT hint, IndexT start, IndexT end)
{
  const T hint_key = search_data[hint];
  if (hint_key == search_key)
  {
    return hint;
  }
  if (hint_key < search_key)
  {
    start = hint;
  }
  else
  {
    end = hint;
  }
  
  return binary_search(search_data, search_key, start, end);
}

} // namespace binfly
