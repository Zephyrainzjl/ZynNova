#include <zynnova/dft/aimd.hpp>
#include <zynnova/dft/quantum.hpp>

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

void test_harmonic_oscillator() {
    constexpr std::size_t point_count = 801;
    constexpr double lower = -8.0;
    constexpr double upper = 8.0;
    std::vector<double> grid(point_count);
    std::vector<double> potential(point_count);
    for (std::size_t i = 0; i < point_count; ++i) {
        grid[i] = lower
            + (upper - lower) * static_cast<double>(i)
                / static_cast<double>(point_count - 1);
        potential[i] = 0.5 * grid[i] * grid[i];
    }
    const auto result = zynnova::dft::solve_schrodinger_1d(
        grid, potential, 1.0, 4, 1.0e-12, 100
    );
    for (std::size_t state = 0; state < 4; ++state) {
        const double exact = static_cast<double>(state) + 0.5;
        require(
            std::abs(result.energies[state] - exact) < 4.0e-4,
            "harmonic-oscillator eigenvalue regression failed"
        );
        require(
            result.residual_norms[state] < 1.0e-7,
            "stationary-state residual is too large"
        );
    }
}

void test_free_particle_velocity_verlet() {
    zynnova::dft::AIMDIntegrator integrator(
        {1.0}, {true}, 0.5, "nve", 300.0, 0.01, 7
    );
    integrator.set_velocities({0.1, -0.2, 0.05});
    const std::vector<double> positions{1.0, 2.0, 3.0};
    const std::vector<double> zero_force{0.0, 0.0, 0.0};
    const auto updated = integrator.begin_step(positions, zero_force);
    const auto final_velocities = integrator.end_step(zero_force);
    require(std::abs(updated[0] - 1.05) < 1.0e-14, "x drift is incorrect");
    require(std::abs(updated[1] - 1.90) < 1.0e-14, "y drift is incorrect");
    require(std::abs(updated[2] - 3.025) < 1.0e-14, "z drift is incorrect");
    require(
        std::abs(final_velocities[0] - 0.1) < 1.0e-14,
        "free-particle velocity changed"
    );
    require(integrator.step_index() == 1, "step counter is incorrect");
}

void test_maxwell_boltzmann_temperature() {
    const std::vector<double> masses{1.0, 12.0, 16.0, 14.0};
    const std::vector<bool> mobile(masses.size(), true);
    const auto velocities = zynnova::dft::maxwell_boltzmann_velocities(
        masses, mobile, 500.0, 19, true, true
    );
    const double temperature = zynnova::dft::instantaneous_temperature_K(
        masses, velocities, mobile, true
    );
    require(std::abs(temperature - 500.0) < 1.0e-10, "temperature rescaling failed");
}

}  // namespace

int main() {
    try {
        test_harmonic_oscillator();
        test_free_particle_velocity_verlet();
        test_maxwell_boltzmann_temperature();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "ZynNova DFT core tests passed\n";
    return EXIT_SUCCESS;
}
