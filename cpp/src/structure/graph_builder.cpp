#include <zynnova/structure/graph_builder.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace zynnova::structure {
namespace {

using Vec3 = std::array<double, 3>;

Vec3 add(const Vec3& a, const Vec3& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

Vec3 sub(const Vec3& a, const Vec3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

Vec3 scale(const Vec3& a, double s) {
    return {a[0] * s, a[1] * s, a[2] * s};
}

double norm(const Vec3& a) {
    return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

Vec3 lattice_translation(
    const std::array<std::array<double, 3>, 3>& cell,
    const std::array<int, 3>& shift
) {
    Vec3 out{0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < 3; ++i) {
        out = add(out, scale(cell[i], static_cast<double>(shift[i])));
    }
    return out;
}

double determinant(const std::array<std::array<double, 3>, 3>& m) {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
         - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
         + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

std::array<std::array<double, 3>, 3> inverse(
    const std::array<std::array<double, 3>, 3>& m
) {
    const double det = determinant(m);
    if (std::abs(det) < 1.0e-14) {
        throw std::invalid_argument("Periodic cell is singular.");
    }
    const double inv_det = 1.0 / det;
    std::array<std::array<double, 3>, 3> out{};
    out[0][0] =  (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
    out[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * inv_det;
    out[0][2] =  (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
    out[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) * inv_det;
    out[1][1] =  (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    out[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) * inv_det;
    out[2][0] =  (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
    out[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * inv_det;
    out[2][2] =  (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
    return out;
}

int image_bound(
    const std::array<std::array<double, 3>, 3>& cell,
    const std::array<bool, 3>& pbc,
    double max_cutoff
) {
    if (!pbc[0] && !pbc[1] && !pbc[2]) {
        return 0;
    }
    const auto inv = inverse(cell);
    double reciprocal_bound = 0.0;
    for (std::size_t col = 0; col < 3; ++col) {
        const Vec3 reciprocal_col{inv[0][col], inv[1][col], inv[2][col]};
        reciprocal_bound = std::max(reciprocal_bound, norm(reciprocal_col));
    }
    return std::max(1, static_cast<int>(std::ceil(max_cutoff * reciprocal_bound)) + 1);
}

bool canonical_undirected(std::size_t source, std::size_t target, const std::array<int, 3>& shift) {
    if (source < target) {
        return true;
    }
    if (source > target) {
        return false;
    }
    return std::tie(shift[0], shift[1], shift[2]) > std::make_tuple(0, 0, 0);
}

}  // namespace

GraphResult build_neighbor_graph(
    const std::vector<std::array<double, 3>>& positions,
    const std::array<std::array<double, 3>, 3>& cell,
    const std::array<bool, 3>& pbc,
    const std::vector<double>& radii,
    const GraphBuildOptions& options
) {
    const std::size_t n = positions.size();
    if (n == 0) {
        return {};
    }
    if (options.mode != "cutoff" && options.mode != "radius" && options.mode != "knn") {
        throw std::invalid_argument("mode must be one of: cutoff, radius, knn");
    }
    if (options.mode == "radius" && radii.size() != n) {
        throw std::invalid_argument("radius mode requires one radius per atom");
    }
    if (options.mode == "knn" && options.max_neighbors == 0) {
        throw std::invalid_argument("knn mode requires max_neighbors > 0");
    }
    const bool any_pbc = pbc[0] || pbc[1] || pbc[2];
    if (options.mode == "knn" && any_pbc && options.cutoff <= 0.0) {
        throw std::invalid_argument("Periodic knn mode requires a positive candidate cutoff");
    }

    double max_cutoff = options.cutoff;
    if (options.mode == "radius") {
        const double max_radius = *std::max_element(radii.begin(), radii.end());
        max_cutoff = 2.0 * options.radius_scale * max_radius;
    }
    if (options.mode != "knn" && max_cutoff <= 0.0) {
        throw std::invalid_argument("cutoff must be positive");
    }

    const int bound = image_bound(cell, pbc, std::max(max_cutoff, 0.0));
    std::vector<EdgeRecord> candidates;

    for (std::size_t source = 0; source < n; ++source) {
        for (std::size_t target = 0; target < n; ++target) {
            const int sx_min = pbc[0] ? -bound : 0;
            const int sx_max = pbc[0] ?  bound : 0;
            const int sy_min = pbc[1] ? -bound : 0;
            const int sy_max = pbc[1] ?  bound : 0;
            const int sz_min = pbc[2] ? -bound : 0;
            const int sz_max = pbc[2] ?  bound : 0;
            for (int sx = sx_min; sx <= sx_max; ++sx) {
                for (int sy = sy_min; sy <= sy_max; ++sy) {
                    for (int sz = sz_min; sz <= sz_max; ++sz) {
                        const std::array<int, 3> shift{sx, sy, sz};
                        const bool zero_self = source == target && sx == 0 && sy == 0 && sz == 0;
                        if (zero_self && !options.self_edges) {
                            continue;
                        }
                        if (!options.directed && !canonical_undirected(source, target, shift)) {
                            continue;
                        }

                        const Vec3 translated = add(positions[target], lattice_translation(cell, shift));
                        const Vec3 vector = sub(translated, positions[source]);
                        const double distance = norm(vector);
                        if (distance <= options.tolerance && !options.self_edges) {
                            continue;
                        }

                        double pair_cutoff = options.cutoff;
                        if (options.mode == "radius") {
                            pair_cutoff = options.radius_scale * (radii[source] + radii[target]);
                        }
                        if ((options.mode == "cutoff" || options.mode == "radius")
                            && distance > pair_cutoff + options.tolerance) {
                            continue;
                        }
                        if (options.mode == "knn" && options.cutoff > 0.0
                            && distance > options.cutoff + options.tolerance) {
                            continue;
                        }
                        candidates.push_back({source, target, shift, vector, distance});
                    }
                }
            }
        }
    }

    std::stable_sort(candidates.begin(), candidates.end(), [](const EdgeRecord& a, const EdgeRecord& b) {
        return std::tie(a.source, a.distance, a.target, a.shift[0], a.shift[1], a.shift[2])
             < std::tie(b.source, b.distance, b.target, b.shift[0], b.shift[1], b.shift[2]);
    });

    if (options.max_neighbors == 0) {
        return {std::move(candidates)};
    }

    std::vector<EdgeRecord> limited;
    limited.reserve(candidates.size());
    std::size_t current_source = std::numeric_limits<std::size_t>::max();
    std::size_t count = 0;
    for (const auto& edge : candidates) {
        if (edge.source != current_source) {
            current_source = edge.source;
            count = 0;
        }
        if (count < options.max_neighbors) {
            limited.push_back(edge);
            ++count;
        }
    }
    return {std::move(limited)};
}

void validate_graph(
    std::size_t num_nodes,
    const std::vector<std::array<std::size_t, 2>>& edges,
    const std::vector<std::array<int, 3>>& shifts
) {
    if (!shifts.empty() && shifts.size() != edges.size()) {
        throw std::invalid_argument("edge_shift length must equal edge count");
    }
    for (const auto& edge : edges) {
        if (edge[0] >= num_nodes || edge[1] >= num_nodes) {
            throw std::out_of_range("edge_index contains a node index outside [0, num_nodes)");
        }
    }
}

}  // namespace zynnova::structure
