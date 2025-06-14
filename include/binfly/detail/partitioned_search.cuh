#pragma once

#include "binfly/detail/binary_search.cuh"
#include <cstdint>
#include <cub/cub.cuh>

namespace binfly
{

template <std::uint32_t TileItems, typename IndexT, typename KeyT>
__global__ void partitioned_search(IndexT* tile_starts,
                                   IndexT num_tiles,
                                   const KeyT* search_keys,
                                   IndexT num_search_keys,
                                   const KeyT* search_data,
                                   IndexT num_search_data)
{
  const IndexT tile_idx = blockIdx.x * TileItems + threadIdx.x;
  if (tile_idx <= num_tiles)
  {
    const IndexT key_idx = cub::min(tile_idx * TileItems, num_search_keys - 1);
    tile_starts[tile_idx] =
      binary_search(search_data, search_keys[key_idx], static_cast<IndexT>(0), num_search_data);
  }
}

} // namespace binfly
