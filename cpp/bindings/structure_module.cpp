#include <zynnova/structure/graph_builder.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using zynnova::structure::GraphBuildOptions;

namespace {

std::vector<std::array<double, 3>> read_positions(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& array
) {
    if (array.ndim() != 2 || array.shape(1) != 3) {
        throw std::invalid_argument("positions must have shape [N, 3]");
    }
    auto view = array.unchecked<2>();
    std::vector<std::array<double, 3>> out(static_cast<std::size_t>(array.shape(0)));
    for (py::ssize_t i = 0; i < array.shape(0); ++i) {
        out[static_cast<std::size_t>(i)] = {view(i, 0), view(i, 1), view(i, 2)};
    }
    return out;
}

std::array<std::array<double, 3>, 3> read_cell(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& array
) {
    if (array.ndim() != 2 || array.shape(0) != 3 || array.shape(1) != 3) {
        throw std::invalid_argument("cell must have shape [3, 3]");
    }
    auto view = array.unchecked<2>();
    std::array<std::array<double, 3>, 3> out{};
    for (py::ssize_t i = 0; i < 3; ++i) {
        for (py::ssize_t j = 0; j < 3; ++j) {
            out[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = view(i, j);
        }
    }
    return out;
}

std::array<bool, 3> read_pbc(
    const py::array_t<bool, py::array::c_style | py::array::forcecast>& array
) {
    if (array.ndim() != 1 || array.shape(0) != 3) {
        throw std::invalid_argument("pbc must have shape [3]");
    }
    auto view = array.unchecked<1>();
    return {view(0), view(1), view(2)};
}

std::vector<double> read_vector(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& array
) {
    if (array.ndim() != 1) {
        throw std::invalid_argument("radii must have shape [N]");
    }
    auto view = array.unchecked<1>();
    std::vector<double> out(static_cast<std::size_t>(array.shape(0)));
    for (py::ssize_t i = 0; i < array.shape(0); ++i) {
        out[static_cast<std::size_t>(i)] = view(i);
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_structure_native, m) {
    m.doc() = "ZynNova native structure-to-graph backend";

    m.def(
        "build_neighbor_graph",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& positions,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& cell,
           const py::array_t<bool, py::array::c_style | py::array::forcecast>& pbc,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& radii,
           const std::string& mode,
           double cutoff,
           double radius_scale,
           std::size_t max_neighbors,
           bool directed,
           bool self_edges,
           double tolerance) {
            GraphBuildOptions options;
            options.mode = mode;
            options.cutoff = cutoff;
            options.radius_scale = radius_scale;
            options.max_neighbors = max_neighbors;
            options.directed = directed;
            options.self_edges = self_edges;
            options.tolerance = tolerance;

            const auto result = zynnova::structure::build_neighbor_graph(
                read_positions(positions), read_cell(cell), read_pbc(pbc), read_vector(radii), options
            );

            const py::ssize_t edge_count = static_cast<py::ssize_t>(result.edges.size());
            py::array_t<long long> edge_index(py::array::ShapeContainer{static_cast<py::ssize_t>(2), edge_count});
            py::array_t<int> edge_shift(py::array::ShapeContainer{edge_count, static_cast<py::ssize_t>(3)});
            py::array_t<double> edge_vec(py::array::ShapeContainer{edge_count, static_cast<py::ssize_t>(3)});
            py::array_t<double> edge_dist({edge_count});
            auto index_out = edge_index.mutable_unchecked<2>();
            auto shift_out = edge_shift.mutable_unchecked<2>();
            auto vector_out = edge_vec.mutable_unchecked<2>();
            auto distance_out = edge_dist.mutable_unchecked<1>();

            for (py::ssize_t e = 0; e < edge_count; ++e) {
                const auto& edge = result.edges[static_cast<std::size_t>(e)];
                index_out(0, e) = static_cast<long long>(edge.source);
                index_out(1, e) = static_cast<long long>(edge.target);
                for (py::ssize_t d = 0; d < 3; ++d) {
                    shift_out(e, d) = edge.shift[static_cast<std::size_t>(d)];
                    vector_out(e, d) = edge.vector[static_cast<std::size_t>(d)];
                }
                distance_out(e) = edge.distance;
            }

            py::dict out;
            out["edge_index"] = std::move(edge_index);
            out["edge_shift"] = std::move(edge_shift);
            out["edge_vec"] = std::move(edge_vec);
            out["edge_dist"] = std::move(edge_dist);
            return out;
        },
        py::arg("positions"),
        py::arg("cell"),
        py::arg("pbc"),
        py::arg("radii"),
        py::arg("mode") = "cutoff",
        py::arg("cutoff") = 5.0,
        py::arg("radius_scale") = 1.2,
        py::arg("max_neighbors") = 0,
        py::arg("directed") = true,
        py::arg("self_edges") = false,
        py::arg("tolerance") = 1.0e-8
    );

    m.def(
        "validate_graph",
        [](std::size_t num_nodes,
           const py::array_t<long long, py::array::c_style | py::array::forcecast>& edge_index,
           const py::array_t<int, py::array::c_style | py::array::forcecast>& edge_shift) {
            if (edge_index.ndim() != 2 || edge_index.shape(0) != 2) {
                throw std::invalid_argument("edge_index must have shape [2, E]");
            }
            if (edge_shift.ndim() != 2 || edge_shift.shape(1) != 3
                || edge_shift.shape(0) != edge_index.shape(1)) {
                throw std::invalid_argument("edge_shift must have shape [E, 3]");
            }
            auto indices = edge_index.unchecked<2>();
            auto shifts_view = edge_shift.unchecked<2>();
            std::vector<std::array<std::size_t, 2>> edges;
            std::vector<std::array<int, 3>> shifts;
            edges.reserve(static_cast<std::size_t>(edge_index.shape(1)));
            shifts.reserve(static_cast<std::size_t>(edge_index.shape(1)));
            for (py::ssize_t e = 0; e < edge_index.shape(1); ++e) {
                if (indices(0, e) < 0 || indices(1, e) < 0) {
                    throw std::out_of_range("edge_index cannot contain negative indices");
                }
                edges.push_back({
                    static_cast<std::size_t>(indices(0, e)),
                    static_cast<std::size_t>(indices(1, e))
                });
                shifts.push_back({shifts_view(e, 0), shifts_view(e, 1), shifts_view(e, 2)});
            }
            zynnova::structure::validate_graph(num_nodes, edges, shifts);
        },
        py::arg("num_nodes"), py::arg("edge_index"), py::arg("edge_shift")
    );
}
