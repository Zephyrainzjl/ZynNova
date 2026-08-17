#include <zynnova/zynsim/fem.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using zynnova::zynsim::COOMatrix;
using zynnova::zynsim::Point3;
using zynnova::zynsim::Tensor3;
using zynnova::zynsim::Tet4;

namespace {

std::vector<Point3> read_nodes(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& values
) {
    if (values.ndim() != 2 || values.shape(1) != 3) {
        throw std::invalid_argument("nodes must have shape (n_nodes, 3)");
    }
    auto view = values.unchecked<2>();
    std::vector<Point3> nodes(static_cast<std::size_t>(values.shape(0)));
    for (py::ssize_t i = 0; i < values.shape(0); ++i) {
        nodes[static_cast<std::size_t>(i)] = {view(i, 0), view(i, 1), view(i, 2)};
    }
    return nodes;
}

std::vector<Tet4> read_cells(
    const py::array_t<long long, py::array::c_style | py::array::forcecast>& values
) {
    if (values.ndim() != 2 || values.shape(1) != 4) {
        throw std::invalid_argument("cells must have shape (n_cells, 4)");
    }
    auto view = values.unchecked<2>();
    std::vector<Tet4> cells(static_cast<std::size_t>(values.shape(0)));
    for (py::ssize_t i = 0; i < values.shape(0); ++i) {
        for (py::ssize_t j = 0; j < 4; ++j) {
            if (view(i, j) < 0) {
                throw std::out_of_range("cell indices cannot be negative");
            }
            cells[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                static_cast<std::size_t>(view(i, j));
        }
    }
    return cells;
}

std::array<Point3, 4> read_element_coordinates(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& values
) {
    if (values.ndim() != 2 || values.shape(0) != 4 || values.shape(1) != 3) {
        throw std::invalid_argument("element coordinates must have shape (4, 3)");
    }
    auto view = values.unchecked<2>();
    std::array<Point3, 4> result{};
    for (py::ssize_t i = 0; i < 4; ++i) {
        result[static_cast<std::size_t>(i)] = {view(i, 0), view(i, 1), view(i, 2)};
    }
    return result;
}

std::vector<double> read_element_scalars(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& values,
    std::size_t element_count,
    const char* name
) {
    if (values.ndim() == 0) {
        return {values.at()};
    }
    if (values.ndim() != 1
        || (values.shape(0) != 1
            && values.shape(0) != static_cast<py::ssize_t>(element_count))) {
        throw std::invalid_argument(
            std::string(name) + " must be scalar or have shape (n_cells,)"
        );
    }
    auto view = values.unchecked<1>();
    std::vector<double> result(static_cast<std::size_t>(values.shape(0)));
    for (py::ssize_t i = 0; i < values.shape(0); ++i) {
        result[static_cast<std::size_t>(i)] = view(i);
    }
    return result;
}

std::vector<Tensor3> read_tensors(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& values,
    std::size_t element_count
) {
    std::vector<Tensor3> result;
    if (values.ndim() == 2 && values.shape(0) == 3 && values.shape(1) == 3) {
        auto view = values.unchecked<2>();
        Tensor3 tensor{};
        for (py::ssize_t i = 0; i < 3; ++i) {
            for (py::ssize_t j = 0; j < 3; ++j) {
                tensor[static_cast<std::size_t>(i * 3 + j)] = view(i, j);
            }
        }
        result.push_back(tensor);
        return result;
    }
    if (values.ndim() != 3
        || values.shape(0) != static_cast<py::ssize_t>(element_count)
        || values.shape(1) != 3 || values.shape(2) != 3) {
        throw std::invalid_argument(
            "conductivity must have shape (3, 3) or (n_cells, 3, 3)"
        );
    }
    auto view = values.unchecked<3>();
    result.resize(element_count);
    for (py::ssize_t e = 0; e < values.shape(0); ++e) {
        for (py::ssize_t i = 0; i < 3; ++i) {
            for (py::ssize_t j = 0; j < 3; ++j) {
                result[static_cast<std::size_t>(e)][static_cast<std::size_t>(i * 3 + j)] =
                    view(e, i, j);
            }
        }
    }
    return result;
}

template <std::size_t Rows, std::size_t Columns>
py::array_t<double> matrix_array(const std::array<double, Rows * Columns>& values) {
    py::array_t<double> output(
        py::array::ShapeContainer{
            static_cast<py::ssize_t>(Rows),
            static_cast<py::ssize_t>(Columns),
        }
    );
    auto view = output.template mutable_unchecked<2>();
    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Columns; ++j) {
            view(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(j)) =
                values[i * Columns + j];
        }
    }
    return output;
}

py::dict coo_dict(const COOMatrix& matrix) {
    py::array_t<long long> rows(matrix.row_indices.size());
    py::array_t<long long> columns(matrix.column_indices.size());
    py::array_t<double> values(matrix.values.size());
    auto row_view = rows.mutable_unchecked<1>();
    auto column_view = columns.mutable_unchecked<1>();
    auto value_view = values.mutable_unchecked<1>();
    for (std::size_t index = 0; index < matrix.values.size(); ++index) {
        row_view(static_cast<py::ssize_t>(index)) =
            static_cast<long long>(matrix.row_indices[index]);
        column_view(static_cast<py::ssize_t>(index)) =
            static_cast<long long>(matrix.column_indices[index]);
        value_view(static_cast<py::ssize_t>(index)) = matrix.values[index];
    }
    py::dict output;
    output["rows"] = std::move(rows);
    output["columns"] = std::move(columns);
    output["values"] = std::move(values);
    output["shape"] = py::make_tuple(matrix.rows, matrix.columns);
    return output;
}

}  // namespace

PYBIND11_MODULE(_zynsim_fem_native, module) {
    module.doc() = "ZynSim tetrahedral finite-element assembly backend";

    module.def(
        "tet4_geometry",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& coordinates) {
            const auto geometry =
                zynnova::zynsim::tet4_geometry(read_element_coordinates(coordinates));
            py::array_t<double> gradients(
                py::array::ShapeContainer{
                    static_cast<py::ssize_t>(4),
                    static_cast<py::ssize_t>(3),
                }
            );
            auto view = gradients.mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < 4; ++i) {
                for (py::ssize_t j = 0; j < 3; ++j) {
                    view(i, j) =
                        geometry.gradients[static_cast<std::size_t>(i)]
                                          [static_cast<std::size_t>(j)];
                }
            }
            py::dict output;
            output["volume"] = geometry.volume;
            output["gradients"] = std::move(gradients);
            return output;
        },
        py::arg("coordinates")
    );

    module.def(
        "tet4_scalar_matrices",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& coordinates,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& conductivity,
           double density,
           bool lumped) {
            const auto element = read_element_coordinates(coordinates);
            const auto tensors = read_tensors(conductivity, 1);
            py::dict output;
            output["mass"] = matrix_array<4, 4>(
                zynnova::zynsim::tet4_scalar_mass(element, density, lumped)
            );
            output["stiffness"] = matrix_array<4, 4>(
                zynnova::zynsim::tet4_scalar_diffusion(element, tensors.front())
            );
            return output;
        },
        py::arg("coordinates"),
        py::arg("conductivity"),
        py::arg("density") = 1.0,
        py::arg("lumped") = false
    );

    module.def(
        "tet4_elastic_stiffness",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& coordinates,
           double young_modulus,
           double poisson_ratio) {
            return matrix_array<12, 12>(
                zynnova::zynsim::tet4_linear_elastic_stiffness(
                    read_element_coordinates(coordinates), young_modulus, poisson_ratio
                )
            );
        },
        py::arg("coordinates"),
        py::arg("young_modulus"),
        py::arg("poisson_ratio")
    );

    module.def(
        "tet4_neo_hookean",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& coordinates,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& displacement,
           double shear_modulus,
           double lame_lambda) {
            if (displacement.ndim() != 2
                || displacement.shape(0) != 4 || displacement.shape(1) != 3) {
                throw std::invalid_argument("displacement must have shape (4, 3)");
            }
            auto view = displacement.unchecked<2>();
            std::array<double, 12> values{};
            for (py::ssize_t i = 0; i < 4; ++i) {
                for (py::ssize_t j = 0; j < 3; ++j) {
                    values[static_cast<std::size_t>(i * 3 + j)] = view(i, j);
                }
            }
            const auto result = zynnova::zynsim::tet4_compressible_neo_hookean(
                read_element_coordinates(coordinates), values, shear_modulus, lame_lambda
            );
            py::array_t<double> residual(
                py::array::ShapeContainer{
                    static_cast<py::ssize_t>(4),
                    static_cast<py::ssize_t>(3),
                }
            );
            auto residual_view = residual.mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < 4; ++i) {
                for (py::ssize_t j = 0; j < 3; ++j) {
                    residual_view(i, j) =
                        result.residual[static_cast<std::size_t>(i * 3 + j)];
                }
            }
            py::dict output;
            output["energy"] = result.energy;
            output["residual"] = std::move(residual);
            output["tangent"] = matrix_array<12, 12>(result.tangent);
            output["jacobian"] = result.jacobian;
            return output;
        },
        py::arg("coordinates"),
        py::arg("displacement"),
        py::arg("shear_modulus"),
        py::arg("lame_lambda")
    );

    module.def(
        "assemble_scalar",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& node_values,
           const py::array_t<long long, py::array::c_style | py::array::forcecast>& cell_values,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& conductivity,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& density,
           bool lumped) {
            const auto nodes = read_nodes(node_values);
            const auto cells = read_cells(cell_values);
            const auto result = zynnova::zynsim::assemble_scalar_tet4(
                nodes,
                cells,
                read_tensors(conductivity, cells.size()),
                read_element_scalars(density, cells.size(), "density"),
                lumped
            );
            py::dict output;
            output["mass"] = coo_dict(result.mass);
            output["stiffness"] = coo_dict(result.stiffness);
            return output;
        },
        py::arg("nodes"),
        py::arg("cells"),
        py::arg("conductivity"),
        py::arg("density"),
        py::arg("lumped") = false
    );

    module.def(
        "assemble_elasticity",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& node_values,
           const py::array_t<long long, py::array::c_style | py::array::forcecast>& cell_values,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& young_modulus,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& poisson_ratio) {
            const auto nodes = read_nodes(node_values);
            const auto cells = read_cells(cell_values);
            return coo_dict(zynnova::zynsim::assemble_linear_elasticity_tet4(
                nodes,
                cells,
                read_element_scalars(young_modulus, cells.size(), "young_modulus"),
                read_element_scalars(poisson_ratio, cells.size(), "poisson_ratio")
            ));
        },
        py::arg("nodes"),
        py::arg("cells"),
        py::arg("young_modulus"),
        py::arg("poisson_ratio")
    );
}
