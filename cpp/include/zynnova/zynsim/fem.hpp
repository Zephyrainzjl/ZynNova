#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace zynnova::zynsim {

using Point3 = std::array<double, 3>;
using Tet4 = std::array<std::size_t, 4>;
using Tensor3 = std::array<double, 9>;

struct Tet4Geometry {
    double volume{};
    std::array<Point3, 4> gradients{};
};

struct COOMatrix {
    std::size_t rows{};
    std::size_t columns{};
    std::vector<std::size_t> row_indices;
    std::vector<std::size_t> column_indices;
    std::vector<double> values;
};

struct ScalarAssembly {
    COOMatrix mass;
    COOMatrix stiffness;
};

struct NeoHookeanResult {
    double energy{};
    std::array<double, 12> residual{};
    std::array<double, 144> tangent{};
    double jacobian{};
};

Tet4Geometry tet4_geometry(const std::array<Point3, 4>& coordinates);

std::array<double, 16> tet4_scalar_mass(
    const std::array<Point3, 4>& coordinates,
    double density,
    bool lumped
);

std::array<double, 16> tet4_scalar_diffusion(
    const std::array<Point3, 4>& coordinates,
    const Tensor3& conductivity
);

std::array<double, 144> tet4_linear_elastic_stiffness(
    const std::array<Point3, 4>& coordinates,
    double young_modulus,
    double poisson_ratio
);

NeoHookeanResult tet4_compressible_neo_hookean(
    const std::array<Point3, 4>& coordinates,
    const std::array<double, 12>& displacement,
    double shear_modulus,
    double lame_lambda
);

ScalarAssembly assemble_scalar_tet4(
    const std::vector<Point3>& nodes,
    const std::vector<Tet4>& cells,
    const std::vector<Tensor3>& conductivity,
    const std::vector<double>& density,
    bool lumped
);

COOMatrix assemble_linear_elasticity_tet4(
    const std::vector<Point3>& nodes,
    const std::vector<Tet4>& cells,
    const std::vector<double>& young_modulus,
    const std::vector<double>& poisson_ratio
);

}  // namespace zynnova::zynsim
