#include "zynnova/zynsim/voxel.hpp"

#include <cassert>
#include <cstdint>
#include <vector>

int main() {
  using namespace zynnova::zynsim;
  const std::vector<std::uint8_t> image{0, 20, 120, 255};
  const auto segmented = threshold_labels_u8(
      image.data(), image.size(), {0, 101}, {100, 255}, {2, 1}, -1);
  assert((segmented == std::vector<std::int32_t>{2, 2, 1, 1}));

  const std::vector<std::int32_t> xy{1, 2, 2, 1};
  const std::vector<std::int32_t> xz{1, 2, 1, 2};
  const std::vector<std::int32_t> yz{1, 2, 2, 1};
  const auto fused = fuse_orthogonal_labels(
      xy.data(), xz.data(), yz.data(), 2, 2, 2, {1, 2}, {0.0, 0.0}, 4.0);
  assert(fused.size() == 8);

  const auto counts = phase_counts(fused.data(), fused.size());
  assert(counts.at(1) + counts.at(2) == 8);
  const auto interfaces = interface_counts(fused.data(), 2, 2, 2);
  assert(interfaces[0] + interfaces[1] + interfaces[2] > 0);

  const auto hexes = hex_connectivity_range(2, 2, 2, 0, 8);
  assert(hexes.size() == 64);
  const std::vector<std::int64_t> first_cell{0, 9, 3, 12, 1, 10, 4, 13};
  assert(std::vector<std::int64_t>(hexes.begin(), hexes.begin() + 8) == first_cell);
  const std::vector<std::int64_t> x_neighbor{9, 18, 12, 21, 10, 19, 13, 22};
  assert(std::vector<std::int64_t>(hexes.begin() + 32, hexes.begin() + 40) == x_neighbor);
  return 0;
}
