#include <zynnova/zynsim/fem.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace zynnova::zynsim {
namespace {

using Matrix3 = std::array<double, 9>;

double determinant(const Matrix3& a) {
    return a[0] * (a[4] * a[8] - a[5] * a[7])
        - a[1] * (a[3] * a[8] - a[5] * a[6])
        + a[2] * (a[3] * a[7] - a[4] * a[6]);
}

Matrix3 inverse(const Matrix3& a, double& determinant_out) {
    determinant_out = determinant(a);
    const double scale = *std::max_element(
        a.begin(), a.end(), [](double left, double right) {
            return std::abs(left) < std::abs(right);
        }
    );
    const double threshold = std::max(
        std::pow(std::abs(scale), 3.0),
        std::numeric_limits<double>::min()
    ) * 64.0 * std::numeric_limits<double>::epsilon();
    if (!std::isfinite(determinant_out) || std::abs(determinant_out) <= threshold) {
        throw std::invalid_argument("degenerate or non-finite tetrahedral mapping");
    }
    Matrix3 result{};
    result[0] = (a[4] * a[8] - a[5] * a[7]) / determinant_out;
    result[1] = (a[2] * a[7] - a[1] * a[8]) / determinant_out;
    result[2] = (a[1] * a[5] - a[2] * a[4]) / determinant_out;
    result[3] = (a[5] * a[6] - a[3] * a[8]) / determinant_out;
    result[4] = (a[0] * a[8] - a[2] * a[6]) / determinant_out;
    result[5] = (a[2] * a[3] - a[0] * a[5]) / determinant_out;
    result[6] = (a[3] * a[7] - a[4] * a[6]) / determinant_out;
    result[7] = (a[1] * a[6] - a[0] * a[7]) / determinant_out;
    result[8] = (a[0] * a[4] - a[1] * a[3]) / determinant_out;
    return result;
}

std::array<Point3, 4> gather(
    const std::vector<Point3>& nodes,
    const Tet4& cell
) {
    std::array<Point3, 4> coordinates{};
    for (std::size_t local = 0; local < 4; ++local) {
        if (cell[local] >= nodes.size()) {
            throw std::out_of_range("tetrahedron contains an invalid node index");
        }
        coordinates[local] = nodes[cell[local]];
    }
    return coordinates;
}

template <typename Value>
const Value& element_value(
    const std::vector<Value>& values,
    std::size_t element,
    std::size_t element_count,
    const char* name
) {
    if (values.size() == 1) {
        return values.front();
    }
    if (values.size() != element_count) {
        throw std::invalid_argument(
            std::string(name) + " must contain one value or one value per cell"
        );
    }
    return values[element];
}

void validate_isotropic(double young_modulus, double poisson_ratio) {
    if (!std::isfinite(young_modulus) || young_modulus <= 0.0) {
        throw std::invalid_argument("Young's modulus must be finite and positive");
    }
    if (!std::isfinite(poisson_ratio) || poisson_ratio <= -1.0 || poisson_ratio >= 0.5) {
        throw std::invalid_argument("Poisson ratio must lie strictly between -1 and 0.5");
    }
}

void validate_tensor(const Tensor3& tensor) {
    if (!std::all_of(tensor.begin(), tensor.end(), [](double value) {
            return std::isfinite(value);
        })) {
        throw std::invalid_argument("conductivity tensor contains a non-finite value");
    }
    double scale = 0.0;
    for (double value : tensor) {
        scale = std::max(scale, std::abs(value));
    }
    const double tolerance = 128.0 * std::numeric_limits<double>::epsilon()
        * std::max(scale, std::numeric_limits<double>::min());
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = i + 1; j < 3; ++j) {
            if (std::abs(tensor[i * 3 + j] - tensor[j * 3 + i]) > tolerance) {
                throw std::invalid_argument("conductivity tensor must be symmetric");
            }
        }
        if (tensor[i * 3 + i] < -tolerance) {
            throw std::invalid_argument("conductivity tensor must be positive semidefinite");
        }
    }
    const double minor01 = tensor[0] * tensor[4] - tensor[1] * tensor[3];
    const double minor02 = tensor[0] * tensor[8] - tensor[2] * tensor[6];
    const double minor12 = tensor[4] * tensor[8] - tensor[5] * tensor[7];
    const double minor_tolerance = tolerance * std::max(scale, 1.0);
    const double determinant_tolerance = minor_tolerance * std::max(scale, 1.0);
    if (minor01 < -minor_tolerance || minor02 < -minor_tolerance
        || minor12 < -minor_tolerance || determinant(tensor) < -determinant_tolerance) {
        throw std::invalid_argument("conductivity tensor must be positive semidefinite");
    }
}

}  // namespace

Tet4Geometry tet4_geometry(const std::array<Point3, 4>& coordinates) {
    Matrix3 jacobian{};
    for (std::size_t physical = 0; physical < 3; ++physical) {
        for (std::size_t reference = 0; reference < 3; ++reference) {
            jacobian[physical * 3 + reference] =
                coordinates[reference + 1][physical] - coordinates[0][physical];
        }
    }
    double jacobian_determinant = 0.0;
    const Matrix3 inverse_jacobian = inverse(jacobian, jacobian_determinant);
    const std::array<Point3, 4> reference_gradients{{
        {-1.0, -1.0, -1.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    }};
    Tet4Geometry result;
    result.volume = std::abs(jacobian_determinant) / 6.0;
    for (std::size_t node = 0; node < 4; ++node) {
        for (std::size_t physical = 0; physical < 3; ++physical) {
            double value = 0.0;
            for (std::size_t reference = 0; reference < 3; ++reference) {
                value += inverse_jacobian[reference * 3 + physical]
                    * reference_gradients[node][reference];
            }
            result.gradients[node][physical] = value;
        }
    }
    return result;
}

std::array<double, 16> tet4_scalar_mass(
    const std::array<Point3, 4>& coordinates,
    double density,
    bool lumped
) {
    if (!std::isfinite(density) || density < 0.0) {
        throw std::invalid_argument("mass density must be finite and non-negative");
    }
    const auto geometry = tet4_geometry(coordinates);
    std::array<double, 16> mass{};
    if (lumped) {
        for (std::size_t i = 0; i < 4; ++i) {
            mass[i * 4 + i] = density * geometry.volume / 4.0;
        }
    } else {
        const double factor = density * geometry.volume / 20.0;
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                mass[i * 4 + j] = factor * (i == j ? 2.0 : 1.0);
            }
        }
    }
    return mass;
}

std::array<double, 16> tet4_scalar_diffusion(
    const std::array<Point3, 4>& coordinates,
    const Tensor3& conductivity
) {
    validate_tensor(conductivity);
    const auto geometry = tet4_geometry(coordinates);
    std::array<double, 16> stiffness{};
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            double value = 0.0;
            for (std::size_t row = 0; row < 3; ++row) {
                for (std::size_t column = 0; column < 3; ++column) {
                    value += geometry.gradients[i][row]
                        * conductivity[row * 3 + column]
                        * geometry.gradients[j][column];
                }
            }
            stiffness[i * 4 + j] = geometry.volume * value;
        }
    }
    return stiffness;
}

std::array<double, 144> tet4_linear_elastic_stiffness(
    const std::array<Point3, 4>& coordinates,
    double young_modulus,
    double poisson_ratio
) {
    validate_isotropic(young_modulus, poisson_ratio);
    const auto geometry = tet4_geometry(coordinates);
    const double shear = young_modulus / (2.0 * (1.0 + poisson_ratio));
    const double lame = young_modulus * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
    std::array<double, 36> constitutive{};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            constitutive[i * 6 + j] = lame + (i == j ? 2.0 * shear : 0.0);
        }
    }
    constitutive[3 * 6 + 3] = shear;
    constitutive[4 * 6 + 4] = shear;
    constitutive[5 * 6 + 5] = shear;

    std::array<double, 72> strain_displacement{};
    for (std::size_t node = 0; node < 4; ++node) {
        const double dx = geometry.gradients[node][0];
        const double dy = geometry.gradients[node][1];
        const double dz = geometry.gradients[node][2];
        const std::size_t column = node * 3;
        strain_displacement[0 * 12 + column + 0] = dx;
        strain_displacement[1 * 12 + column + 1] = dy;
        strain_displacement[2 * 12 + column + 2] = dz;
        strain_displacement[3 * 12 + column + 0] = dy;
        strain_displacement[3 * 12 + column + 1] = dx;
        strain_displacement[4 * 12 + column + 1] = dz;
        strain_displacement[4 * 12 + column + 2] = dy;
        strain_displacement[5 * 12 + column + 0] = dz;
        strain_displacement[5 * 12 + column + 2] = dx;
    }

    std::array<double, 144> stiffness{};
    for (std::size_t row = 0; row < 12; ++row) {
        for (std::size_t column = 0; column < 12; ++column) {
            double value = 0.0;
            for (std::size_t alpha = 0; alpha < 6; ++alpha) {
                for (std::size_t beta = 0; beta < 6; ++beta) {
                    value += strain_displacement[alpha * 12 + row]
                        * constitutive[alpha * 6 + beta]
                        * strain_displacement[beta * 12 + column];
                }
            }
            stiffness[row * 12 + column] = geometry.volume * value;
        }
    }
    return stiffness;
}

NeoHookeanResult tet4_compressible_neo_hookean(
    const std::array<Point3, 4>& coordinates,
    const std::array<double, 12>& displacement,
    double shear_modulus,
    double lame_lambda
) {
    if (!std::isfinite(shear_modulus) || shear_modulus <= 0.0
        || !std::isfinite(lame_lambda) || lame_lambda < 0.0) {
        throw std::invalid_argument("Neo-Hookean moduli must be finite and non-negative");
    }
    const auto geometry = tet4_geometry(coordinates);
    Matrix3 deformation{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    for (std::size_t node = 0; node < 4; ++node) {
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t reference = 0; reference < 3; ++reference) {
                deformation[i * 3 + reference] += displacement[node * 3 + i]
                    * geometry.gradients[node][reference];
            }
        }
    }
    double jacobian = 0.0;
    const Matrix3 inverse_deformation = inverse(deformation, jacobian);
    if (jacobian <= 0.0) {
        throw std::domain_error("Neo-Hookean element has a non-positive deformation Jacobian");
    }
    Matrix3 inverse_transpose{};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            inverse_transpose[i * 3 + j] = inverse_deformation[j * 3 + i];
        }
    }
    const double log_j = std::log(jacobian);
    double first_invariant = 0.0;
    for (double value : deformation) {
        first_invariant += value * value;
    }

    Matrix3 first_piola{};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            const std::size_t index = i * 3 + j;
            first_piola[index] = shear_modulus
                * (deformation[index] - inverse_transpose[index])
                + lame_lambda * log_j * inverse_transpose[index];
        }
    }

    NeoHookeanResult result;
    result.energy = geometry.volume
        * (0.5 * shear_modulus * (first_invariant - 3.0)
           - shear_modulus * log_j + 0.5 * lame_lambda * log_j * log_j);
    result.jacobian = jacobian;
    for (std::size_t a = 0; a < 4; ++a) {
        for (std::size_t i = 0; i < 3; ++i) {
            double value = 0.0;
            for (std::size_t reference = 0; reference < 3; ++reference) {
                value += first_piola[i * 3 + reference]
                    * geometry.gradients[a][reference];
            }
            result.residual[a * 3 + i] = geometry.volume * value;
        }
    }

    for (std::size_t a = 0; a < 4; ++a) {
        for (std::size_t i = 0; i < 3; ++i) {
            const std::size_t row = a * 3 + i;
            for (std::size_t b = 0; b < 4; ++b) {
                for (std::size_t k = 0; k < 3; ++k) {
                    const std::size_t column = b * 3 + k;
                    double value = 0.0;
                    for (std::size_t reference_j = 0; reference_j < 3; ++reference_j) {
                        for (std::size_t reference_l = 0; reference_l < 3; ++reference_l) {
                            const double delta = (i == k && reference_j == reference_l)
                                ? shear_modulus
                                : 0.0;
                            const double inverse_product =
                                inverse_transpose[i * 3 + reference_l]
                                * inverse_transpose[k * 3 + reference_j];
                            const double volumetric =
                                inverse_transpose[k * 3 + reference_l]
                                * inverse_transpose[i * 3 + reference_j];
                            const double tangent = delta
                                + (shear_modulus - lame_lambda * log_j) * inverse_product
                                + lame_lambda * volumetric;
                            value += geometry.gradients[a][reference_j] * tangent
                                * geometry.gradients[b][reference_l];
                        }
                    }
                    result.tangent[row * 12 + column] = geometry.volume * value;
                }
            }
        }
    }
    return result;
}

ScalarAssembly assemble_scalar_tet4(
    const std::vector<Point3>& nodes,
    const std::vector<Tet4>& cells,
    const std::vector<Tensor3>& conductivity,
    const std::vector<double>& density,
    bool lumped
) {
    if (conductivity.empty() || density.empty()) {
        throw std::invalid_argument("conductivity and density cannot be empty");
    }
    if ((conductivity.size() != 1 && conductivity.size() != cells.size())
        || (density.size() != 1 && density.size() != cells.size())) {
        throw std::invalid_argument(
            "conductivity and density must be scalar or contain one value per cell"
        );
    }
    for (std::size_t element = 0; element < cells.size(); ++element) {
        const auto coordinates = gather(nodes, cells[element]);
        static_cast<void>(tet4_geometry(coordinates));
        validate_tensor(element_value(
            conductivity, element, cells.size(), "conductivity"
        ));
        const double value = element_value(density, element, cells.size(), "density");
        if (!std::isfinite(value) || value < 0.0) {
            throw std::invalid_argument("density must be finite and non-negative");
        }
    }
    const std::size_t entry_count = cells.size() * 16;
    ScalarAssembly result;
    for (COOMatrix* matrix : {&result.mass, &result.stiffness}) {
        matrix->rows = nodes.size();
        matrix->columns = nodes.size();
        matrix->row_indices.resize(entry_count);
        matrix->column_indices.resize(entry_count);
        matrix->values.resize(entry_count);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t signed_element = 0;
         signed_element < static_cast<std::int64_t>(cells.size());
         ++signed_element) {
        const auto element = static_cast<std::size_t>(signed_element);
        const auto coordinates = gather(nodes, cells[element]);
        const auto local_mass = tet4_scalar_mass(
            coordinates,
            element_value(density, element, cells.size(), "density"),
            lumped
        );
        const auto local_stiffness = tet4_scalar_diffusion(
            coordinates,
            element_value(conductivity, element, cells.size(), "conductivity")
        );
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                const std::size_t local = i * 4 + j;
                const std::size_t global = element * 16 + local;
                result.mass.row_indices[global] = cells[element][i];
                result.mass.column_indices[global] = cells[element][j];
                result.mass.values[global] = local_mass[local];
                result.stiffness.row_indices[global] = cells[element][i];
                result.stiffness.column_indices[global] = cells[element][j];
                result.stiffness.values[global] = local_stiffness[local];
            }
        }
    }
    return result;
}

COOMatrix assemble_linear_elasticity_tet4(
    const std::vector<Point3>& nodes,
    const std::vector<Tet4>& cells,
    const std::vector<double>& young_modulus,
    const std::vector<double>& poisson_ratio
) {
    if (young_modulus.empty() || poisson_ratio.empty()) {
        throw std::invalid_argument("elastic material arrays cannot be empty");
    }
    if ((young_modulus.size() != 1 && young_modulus.size() != cells.size())
        || (poisson_ratio.size() != 1 && poisson_ratio.size() != cells.size())) {
        throw std::invalid_argument(
            "elastic parameters must be scalar or contain one value per cell"
        );
    }
    for (std::size_t element = 0; element < cells.size(); ++element) {
        static_cast<void>(tet4_geometry(gather(nodes, cells[element])));
        validate_isotropic(
            element_value(young_modulus, element, cells.size(), "young_modulus"),
            element_value(poisson_ratio, element, cells.size(), "poisson_ratio")
        );
    }
    COOMatrix result;
    result.rows = nodes.size() * 3;
    result.columns = nodes.size() * 3;
    const std::size_t entry_count = cells.size() * 144;
    result.row_indices.resize(entry_count);
    result.column_indices.resize(entry_count);
    result.values.resize(entry_count);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t signed_element = 0;
         signed_element < static_cast<std::int64_t>(cells.size());
         ++signed_element) {
        const auto element = static_cast<std::size_t>(signed_element);
        const auto coordinates = gather(nodes, cells[element]);
        const auto local_stiffness = tet4_linear_elastic_stiffness(
            coordinates,
            element_value(young_modulus, element, cells.size(), "young_modulus"),
            element_value(poisson_ratio, element, cells.size(), "poisson_ratio")
        );
        for (std::size_t local_row = 0; local_row < 12; ++local_row) {
            const std::size_t node_row = local_row / 3;
            const std::size_t component_row = local_row % 3;
            for (std::size_t local_column = 0; local_column < 12; ++local_column) {
                const std::size_t node_column = local_column / 3;
                const std::size_t component_column = local_column % 3;
                const std::size_t global = element * 144 + local_row * 12 + local_column;
                result.row_indices[global] =
                    cells[element][node_row] * 3 + component_row;
                result.column_indices[global] =
                    cells[element][node_column] * 3 + component_column;
                result.values[global] =
                    local_stiffness[local_row * 12 + local_column];
            }
        }
    }
    return result;
}

}  // namespace zynnova::zynsim
