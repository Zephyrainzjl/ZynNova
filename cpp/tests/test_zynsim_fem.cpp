#include <zynnova/zynsim/fem.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

const std::array<zynnova::zynsim::Point3, 4> unit_tetra{{
    {0.0, 0.0, 0.0},
    {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {0.0, 0.0, 1.0},
}};

void test_geometry_and_scalar_operators() {
    const auto geometry = zynnova::zynsim::tet4_geometry(unit_tetra);
    require(std::abs(geometry.volume - 1.0 / 6.0) < 1.0e-15, "tetrahedron volume failed");
    for (std::size_t component = 0; component < 3; ++component) {
        double sum = 0.0;
        for (const auto& gradient : geometry.gradients) {
            sum += gradient[component];
        }
        require(std::abs(sum) < 1.0e-15, "partition-of-unity gradient failed");
    }

    const auto mass = zynnova::zynsim::tet4_scalar_mass(unit_tetra, 2.0, false);
    double mass_sum = 0.0;
    for (double value : mass) {
        mass_sum += value;
    }
    require(std::abs(mass_sum - 1.0 / 3.0) < 1.0e-15, "consistent mass failed");

    const zynnova::zynsim::Tensor3 identity{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    };
    const auto stiffness =
        zynnova::zynsim::tet4_scalar_diffusion(unit_tetra, identity);
    for (std::size_t row = 0; row < 4; ++row) {
        double row_sum = 0.0;
        for (std::size_t column = 0; column < 4; ++column) {
            row_sum += stiffness[row * 4 + column];
            require(
                std::abs(stiffness[row * 4 + column] - stiffness[column * 4 + row])
                    < 1.0e-15,
                "diffusion matrix lost symmetry"
            );
        }
        require(std::abs(row_sum) < 1.0e-15, "diffusion constant nullspace failed");
    }

    auto nanometer_tetra = unit_tetra;
    for (auto& point : nanometer_tetra) {
        for (double& coordinate : point) {
            coordinate *= 1.0e-9;
        }
    }
    const auto nanometer_geometry = zynnova::zynsim::tet4_geometry(nanometer_tetra);
    require(
        std::abs(nanometer_geometry.volume - 1.0e-27 / 6.0) < 1.0e-42,
        "scale-relative geometry tolerance failed"
    );
}

void test_elasticity_rigid_mode() {
    const auto stiffness =
        zynnova::zynsim::tet4_linear_elastic_stiffness(unit_tetra, 210.0e9, 0.3);
    for (std::size_t row = 0; row < 12; ++row) {
        for (std::size_t column = 0; column < 12; ++column) {
            require(
                std::abs(stiffness[row * 12 + column] - stiffness[column * 12 + row])
                    < 1.0e-3,
                "elastic matrix lost symmetry"
            );
        }
        for (std::size_t component = 0; component < 3; ++component) {
            double translation_action = 0.0;
            for (std::size_t node = 0; node < 4; ++node) {
                translation_action += stiffness[row * 12 + node * 3 + component];
            }
            require(
                std::abs(translation_action) < 1.0e-4,
                "elastic rigid-translation mode failed"
            );
        }
    }
}

void test_neo_hookean_consistent_tangent() {
    std::array<double, 12> displacement{};
    displacement[3] = 0.02;
    displacement[7] = -0.01;
    displacement[11] = 0.015;
    const double mu = 3.2;
    const double lambda = 5.7;
    const auto base = zynnova::zynsim::tet4_compressible_neo_hookean(
        unit_tetra, displacement, mu, lambda
    );
    require(base.energy > 0.0 && base.jacobian > 0.0, "Neo-Hookean state is invalid");
    constexpr double step = 1.0e-7;
    for (std::size_t column = 0; column < 12; ++column) {
        auto plus_u = displacement;
        auto minus_u = displacement;
        plus_u[column] += step;
        minus_u[column] -= step;
        const auto plus = zynnova::zynsim::tet4_compressible_neo_hookean(
            unit_tetra, plus_u, mu, lambda
        );
        const auto minus = zynnova::zynsim::tet4_compressible_neo_hookean(
            unit_tetra, minus_u, mu, lambda
        );
        for (std::size_t row = 0; row < 12; ++row) {
            const double numerical = (plus.residual[row] - minus.residual[row]) / (2.0 * step);
            require(
                std::abs(numerical - base.tangent[row * 12 + column]) < 2.0e-7,
                "Neo-Hookean consistent tangent regression failed"
            );
        }
    }
}

}  // namespace

int main() {
    try {
        test_geometry_and_scalar_operators();
        test_elasticity_rigid_mode();
        test_neo_hookean_consistent_tangent();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "ZynSim FEM core tests passed\n";
    return EXIT_SUCCESS;
}
