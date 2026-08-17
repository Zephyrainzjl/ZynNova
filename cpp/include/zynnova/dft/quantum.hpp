#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace zynnova::dft {

struct StationaryResult {
    std::size_t num_points{};
    std::size_t num_states{};
    std::vector<double> energies;
    std::vector<double> wavefunctions;
    std::vector<double> residual_norms;
};

struct PropagationResult {
    std::size_t num_points{};
    std::size_t num_frames{};
    std::vector<double> times;
    std::vector<std::complex<double>> wavefunctions;
    std::vector<double> norms;
};

StationaryResult solve_schrodinger_1d(
    const std::vector<double>& grid,
    const std::vector<double>& potential,
    double mass,
    std::size_t num_states,
    double tolerance = 1.0e-12,
    std::size_t max_iterations = 80
);

PropagationResult propagate_schrodinger_1d(
    const std::vector<double>& grid,
    const std::vector<double>& potential,
    const std::vector<std::complex<double>>& initial_wavefunction,
    double mass,
    double timestep,
    std::size_t steps,
    std::size_t save_every = 1
);

}  // namespace zynnova::dft
