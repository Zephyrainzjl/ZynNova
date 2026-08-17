#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace zynnova::structure {

struct GraphBuildOptions {
    std::string mode{"cutoff"};
    double cutoff{5.0};
    double radius_scale{1.2};
    std::size_t max_neighbors{0};
    bool directed{true};
    bool self_edges{false};
    double tolerance{1.0e-8};
};

struct EdgeRecord {
    std::size_t source{};
    std::size_t target{};
    std::array<int, 3> shift{};
    std::array<double, 3> vector{};
    double distance{};
};

struct GraphResult {
    std::vector<EdgeRecord> edges;
};

GraphResult build_neighbor_graph(
    const std::vector<std::array<double, 3>>& positions,
    const std::array<std::array<double, 3>, 3>& cell,
    const std::array<bool, 3>& pbc,
    const std::vector<double>& radii,
    const GraphBuildOptions& options
);

void validate_graph(
    std::size_t num_nodes,
    const std::vector<std::array<std::size_t, 2>>& edges,
    const std::vector<std::array<int, 3>>& shifts
);

}  // namespace zynnova::structure
