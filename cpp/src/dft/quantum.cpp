#include <zynnova/dft/quantum.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace zynnova::dft {
namespace {

struct TridiagonalHamiltonian {
    double spacing{};
    std::vector<double> diagonal;
    std::vector<double> off_diagonal;
};

void require_finite(const std::vector<double>& values, const char* name) {
    if (!std::all_of(values.begin(), values.end(), [](double value) {
            return std::isfinite(value);
        })) {
        throw std::invalid_argument(std::string(name) + " must contain only finite values");
    }
}

TridiagonalHamiltonian discretize(
    const std::vector<double>& grid,
    const std::vector<double>& potential,
    double mass
) {
    if (grid.size() < 4) {
        throw std::invalid_argument("grid must contain at least four points");
    }
    if (potential.size() != grid.size()) {
        throw std::invalid_argument("potential must have the same length as grid");
    }
    if (!std::isfinite(mass) || mass <= 0.0) {
        throw std::invalid_argument("mass must be finite and positive");
    }
    require_finite(grid, "grid");
    require_finite(potential, "potential");

    const double spacing = grid[1] - grid[0];
    if (!(spacing > 0.0) || !std::isfinite(spacing)) {
        throw std::invalid_argument("grid must be strictly increasing");
    }
    const double spacing_tolerance =
        64.0 * std::numeric_limits<double>::epsilon()
        * std::max({1.0, std::abs(grid.front()), std::abs(grid.back())});
    for (std::size_t i = 2; i < grid.size(); ++i) {
        const double local_spacing = grid[i] - grid[i - 1];
        if (!(local_spacing > 0.0)
            || std::abs(local_spacing - spacing)
                > spacing_tolerance + 1.0e-10 * std::abs(spacing)) {
            throw std::invalid_argument("grid must be uniformly spaced and strictly increasing");
        }
    }

    const std::size_t interior_size = grid.size() - 2;
    const double kinetic = 1.0 / (2.0 * mass * spacing * spacing);
    TridiagonalHamiltonian hamiltonian;
    hamiltonian.spacing = spacing;
    hamiltonian.diagonal.resize(interior_size);
    hamiltonian.off_diagonal.assign(interior_size - 1, -kinetic);
    for (std::size_t i = 0; i < interior_size; ++i) {
        hamiltonian.diagonal[i] = potential[i + 1] + 2.0 * kinetic;
    }
    return hamiltonian;
}

double matrix_scale(
    const std::vector<double>& diagonal,
    const std::vector<double>& off_diagonal
) {
    double scale = 1.0;
    for (double value : diagonal) {
        scale = std::max(scale, std::abs(value));
    }
    for (double value : off_diagonal) {
        scale = std::max(scale, std::abs(value));
    }
    return scale;
}

std::size_t sturm_count(
    const std::vector<double>& diagonal,
    const std::vector<double>& off_diagonal,
    double value,
    double pivot_floor
) {
    double pivot = diagonal.front() - value;
    if (std::abs(pivot) < pivot_floor) {
        pivot = -pivot_floor;
    }
    std::size_t count = pivot < 0.0 ? 1U : 0U;
    for (std::size_t i = 1; i < diagonal.size(); ++i) {
        pivot = diagonal[i] - value
            - off_diagonal[i - 1] * off_diagonal[i - 1] / pivot;
        if (std::abs(pivot) < pivot_floor) {
            pivot = -pivot_floor;
        }
        if (pivot < 0.0) {
            ++count;
        }
    }
    return count;
}

std::pair<double, double> eigenvalue_bounds(
    const std::vector<double>& diagonal,
    const std::vector<double>& off_diagonal
) {
    double lower = std::numeric_limits<double>::infinity();
    double upper = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < diagonal.size(); ++i) {
        double radius = 0.0;
        if (i > 0) {
            radius += std::abs(off_diagonal[i - 1]);
        }
        if (i + 1 < diagonal.size()) {
            radius += std::abs(off_diagonal[i]);
        }
        lower = std::min(lower, diagonal[i] - radius);
        upper = std::max(upper, diagonal[i] + radius);
    }
    const double padding =
        32.0 * std::numeric_limits<double>::epsilon()
        * std::max({1.0, std::abs(lower), std::abs(upper)});
    return {lower - padding, upper + padding};
}

std::vector<double> lowest_eigenvalues(
    const std::vector<double>& diagonal,
    const std::vector<double>& off_diagonal,
    std::size_t num_states,
    double tolerance,
    std::size_t max_iterations
) {
    const auto [global_lower, global_upper] =
        eigenvalue_bounds(diagonal, off_diagonal);
    const double scale = matrix_scale(diagonal, off_diagonal);
    const double pivot_floor =
        std::max(std::numeric_limits<double>::min() * scale, 1.0e-300);
    std::vector<double> eigenvalues(num_states);

#if defined(_OPENMP)
#pragma omp parallel for if(num_states > 2)
#endif
    for (std::ptrdiff_t state_index = 0;
         state_index < static_cast<std::ptrdiff_t>(num_states);
         ++state_index) {
        const std::size_t state = static_cast<std::size_t>(state_index);
        double lower = global_lower;
        double upper = global_upper;
        for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
            const double midpoint = lower + 0.5 * (upper - lower);
            const std::size_t count =
                sturm_count(diagonal, off_diagonal, midpoint, pivot_floor);
            if (count <= state) {
                lower = midpoint;
            } else {
                upper = midpoint;
            }
            const double width = upper - lower;
            if (width <= tolerance * (1.0 + std::max(std::abs(lower), std::abs(upper)))) {
                break;
            }
        }
        eigenvalues[state] = lower + 0.5 * (upper - lower);
    }
    return eigenvalues;
}

double dot(const std::vector<double>& left, const std::vector<double>& right) {
    return std::inner_product(left.begin(), left.end(), right.begin(), 0.0);
}

double normalize(std::vector<double>& values) {
    const double norm = std::sqrt(std::max(dot(values, values), 0.0));
    if (!(norm > 0.0) || !std::isfinite(norm)) {
        return norm;
    }
    for (double& value : values) {
        value /= norm;
    }
    return norm;
}

void orthogonalize(
    std::vector<double>& vector,
    const std::vector<std::vector<double>>& basis
) {
    for (int pass = 0; pass < 2; ++pass) {
        for (const auto& previous : basis) {
            const double projection = dot(vector, previous);
            for (std::size_t i = 0; i < vector.size(); ++i) {
                vector[i] -= projection * previous[i];
            }
        }
    }
}

std::vector<double> solve_shifted_tridiagonal(
    const std::vector<double>& diagonal,
    const std::vector<double>& off_diagonal,
    double shift,
    const std::vector<double>& right_hand_side
) {
    const std::size_t size = diagonal.size();
    const double scale = matrix_scale(diagonal, off_diagonal);
    const double pivot_floor =
        128.0 * std::numeric_limits<double>::epsilon() * scale;
    std::vector<double> modified_upper(size > 1 ? size - 1 : 0);
    std::vector<double> modified_rhs(size);

    auto safe_pivot = [pivot_floor](double value) {
        if (std::abs(value) >= pivot_floor) {
            return value;
        }
        return std::copysign(pivot_floor, value == 0.0 ? 1.0 : value);
    };

    double pivot = safe_pivot(diagonal[0] - shift);
    if (size > 1) {
        modified_upper[0] = off_diagonal[0] / pivot;
    }
    modified_rhs[0] = right_hand_side[0] / pivot;
    for (std::size_t i = 1; i < size; ++i) {
        pivot = safe_pivot(
            diagonal[i] - shift - off_diagonal[i - 1] * modified_upper[i - 1]
        );
        if (i + 1 < size) {
            modified_upper[i] = off_diagonal[i] / pivot;
        }
        modified_rhs[i] =
            (right_hand_side[i] - off_diagonal[i - 1] * modified_rhs[i - 1]) / pivot;
    }

    std::vector<double> solution(size);
    solution.back() = modified_rhs.back();
    for (std::size_t i = size - 1; i-- > 0;) {
        solution[i] = modified_rhs[i] - modified_upper[i] * solution[i + 1];
    }
    return solution;
}

std::vector<double> apply_tridiagonal(
    const std::vector<double>& diagonal,
    const std::vector<double>& off_diagonal,
    const std::vector<double>& vector
) {
    std::vector<double> result(vector.size());
    for (std::size_t i = 0; i < vector.size(); ++i) {
        result[i] = diagonal[i] * vector[i];
        if (i > 0) {
            result[i] += off_diagonal[i - 1] * vector[i - 1];
        }
        if (i + 1 < vector.size()) {
            result[i] += off_diagonal[i] * vector[i + 1];
        }
    }
    return result;
}

double residual_norm(
    const std::vector<double>& diagonal,
    const std::vector<double>& off_diagonal,
    const std::vector<double>& eigenvector,
    double eigenvalue
) {
    auto residual = apply_tridiagonal(diagonal, off_diagonal, eigenvector);
    for (std::size_t i = 0; i < residual.size(); ++i) {
        residual[i] -= eigenvalue * eigenvector[i];
    }
    return std::sqrt(std::max(dot(residual, residual), 0.0));
}

std::vector<std::vector<double>> inverse_iteration(
    const std::vector<double>& diagonal,
    const std::vector<double>& off_diagonal,
    const std::vector<double>& eigenvalues,
    double tolerance,
    std::size_t max_iterations
) {
    const std::size_t size = diagonal.size();
    const double pi = std::acos(-1.0);
    const double scale = matrix_scale(diagonal, off_diagonal);
    const double shift_offset = std::max(
        64.0 * std::numeric_limits<double>::epsilon() * scale,
        tolerance * scale * 0.1
    );
    std::vector<std::vector<double>> eigenvectors;
    eigenvectors.reserve(eigenvalues.size());

    for (std::size_t state = 0; state < eigenvalues.size(); ++state) {
        std::vector<double> vector(size);
        for (std::size_t i = 0; i < size; ++i) {
            vector[i] = std::sin(
                static_cast<double>((state + 1) * (i + 1)) * pi
                / static_cast<double>(size + 1)
            );
            vector[i] += 0.013 * std::cos(
                static_cast<double>((state + 2) * (i + 1)) * pi
                / static_cast<double>(size + 1)
            );
        }
        orthogonalize(vector, eigenvectors);
        if (!(normalize(vector) > 0.0)) {
            vector.assign(size, 0.0);
            vector[state % size] = 1.0;
        }

        const double shift = eigenvalues[state] - shift_offset;
        const std::size_t iterations = std::max<std::size_t>(
            4, std::min<std::size_t>(max_iterations, 40)
        );
        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            auto next =
                solve_shifted_tridiagonal(diagonal, off_diagonal, shift, vector);
            orthogonalize(next, eigenvectors);
            if (!(normalize(next) > 0.0)) {
                throw std::runtime_error("inverse iteration produced a zero eigenvector");
            }
            const double alignment = std::abs(dot(next, vector));
            vector = std::move(next);
            if (1.0 - alignment <= std::sqrt(tolerance)) {
                break;
            }
        }
        const auto largest = std::max_element(
            vector.begin(), vector.end(), [](double left, double right) {
                return std::abs(left) < std::abs(right);
            }
        );
        if (largest != vector.end() && *largest < 0.0) {
            for (double& value : vector) {
                value = -value;
            }
        }
        eigenvectors.push_back(std::move(vector));
    }
    return eigenvectors;
}

std::vector<std::complex<double>> solve_complex_tridiagonal(
    const std::vector<std::complex<double>>& diagonal,
    const std::vector<std::complex<double>>& off_diagonal,
    const std::vector<std::complex<double>>& right_hand_side
) {
    const std::size_t size = diagonal.size();
    std::vector<std::complex<double>> modified_upper(size > 1 ? size - 1 : 0);
    std::vector<std::complex<double>> modified_rhs(size);
    const double pivot_floor = 64.0 * std::numeric_limits<double>::epsilon();

    auto safe_pivot = [pivot_floor](std::complex<double> value) {
        if (std::abs(value) >= pivot_floor) {
            return value;
        }
        return value + std::complex<double>(pivot_floor, pivot_floor);
    };

    auto pivot = safe_pivot(diagonal.front());
    if (size > 1) {
        modified_upper[0] = off_diagonal[0] / pivot;
    }
    modified_rhs[0] = right_hand_side[0] / pivot;
    for (std::size_t i = 1; i < size; ++i) {
        pivot = safe_pivot(diagonal[i] - off_diagonal[i - 1] * modified_upper[i - 1]);
        if (i + 1 < size) {
            modified_upper[i] = off_diagonal[i] / pivot;
        }
        modified_rhs[i] =
            (right_hand_side[i] - off_diagonal[i - 1] * modified_rhs[i - 1]) / pivot;
    }

    std::vector<std::complex<double>> solution(size);
    solution.back() = modified_rhs.back();
    for (std::size_t i = size - 1; i-- > 0;) {
        solution[i] = modified_rhs[i] - modified_upper[i] * solution[i + 1];
    }
    return solution;
}

double continuous_norm(
    const std::vector<std::complex<double>>& wavefunction,
    double spacing
) {
    double sum = 0.0;
    for (const auto& value : wavefunction) {
        sum += std::norm(value);
    }
    return spacing * sum;
}

}  // namespace

StationaryResult solve_schrodinger_1d(
    const std::vector<double>& grid,
    const std::vector<double>& potential,
    double mass,
    std::size_t num_states,
    double tolerance,
    std::size_t max_iterations
) {
    if (!std::isfinite(tolerance) || tolerance <= 0.0) {
        throw std::invalid_argument("tolerance must be finite and positive");
    }
    if (max_iterations < 8) {
        throw std::invalid_argument("max_iterations must be at least 8");
    }
    const auto hamiltonian = discretize(grid, potential, mass);
    if (num_states == 0 || num_states > hamiltonian.diagonal.size()) {
        throw std::invalid_argument(
            "num_states must lie between one and the number of interior grid points"
        );
    }

    auto energies = lowest_eigenvalues(
        hamiltonian.diagonal,
        hamiltonian.off_diagonal,
        num_states,
        tolerance,
        max_iterations
    );
    auto interior_wavefunctions = inverse_iteration(
        hamiltonian.diagonal,
        hamiltonian.off_diagonal,
        energies,
        tolerance,
        max_iterations
    );

    StationaryResult result;
    result.num_points = grid.size();
    result.num_states = num_states;
    result.energies = std::move(energies);
    result.wavefunctions.assign(num_states * grid.size(), 0.0);
    result.residual_norms.resize(num_states);

    const double continuous_scale = 1.0 / std::sqrt(hamiltonian.spacing);
    for (std::size_t state = 0; state < num_states; ++state) {
        result.residual_norms[state] = residual_norm(
            hamiltonian.diagonal,
            hamiltonian.off_diagonal,
            interior_wavefunctions[state],
            result.energies[state]
        );
        for (std::size_t i = 0; i < interior_wavefunctions[state].size(); ++i) {
            result.wavefunctions[state * grid.size() + i + 1] =
                continuous_scale * interior_wavefunctions[state][i];
        }
    }
    return result;
}

PropagationResult propagate_schrodinger_1d(
    const std::vector<double>& grid,
    const std::vector<double>& potential,
    const std::vector<std::complex<double>>& initial_wavefunction,
    double mass,
    double timestep,
    std::size_t steps,
    std::size_t save_every
) {
    if (!std::isfinite(timestep) || timestep <= 0.0) {
        throw std::invalid_argument("timestep must be finite and positive");
    }
    if (save_every == 0) {
        throw std::invalid_argument("save_every must be positive");
    }
    if (initial_wavefunction.size() != grid.size()) {
        throw std::invalid_argument(
            "initial_wavefunction must have the same length as grid"
        );
    }
    if (!std::all_of(
            initial_wavefunction.begin(),
            initial_wavefunction.end(),
            [](const std::complex<double>& value) {
                return std::isfinite(value.real()) && std::isfinite(value.imag());
            }
        )) {
        throw std::invalid_argument("initial_wavefunction must contain finite values");
    }

    const auto hamiltonian = discretize(grid, potential, mass);
    const std::size_t interior_size = hamiltonian.diagonal.size();
    std::vector<std::complex<double>> wavefunction(interior_size);
    for (std::size_t i = 0; i < interior_size; ++i) {
        wavefunction[i] = initial_wavefunction[i + 1];
    }
    const double initial_norm = continuous_norm(wavefunction, hamiltonian.spacing);
    if (!(initial_norm > 0.0) || !std::isfinite(initial_norm)) {
        throw std::invalid_argument("initial_wavefunction has zero or invalid norm");
    }
    const double normalization = 1.0 / std::sqrt(initial_norm);
    for (auto& value : wavefunction) {
        value *= normalization;
    }

    const std::complex<double> imaginary_unit(0.0, 1.0);
    std::vector<std::complex<double>> left_diagonal(interior_size);
    std::vector<std::complex<double>> left_off(interior_size - 1);
    for (std::size_t i = 0; i < interior_size; ++i) {
        left_diagonal[i] =
            1.0 + 0.5 * imaginary_unit * timestep * hamiltonian.diagonal[i];
    }
    for (std::size_t i = 0; i + 1 < interior_size; ++i) {
        left_off[i] =
            0.5 * imaginary_unit * timestep * hamiltonian.off_diagonal[i];
    }

    PropagationResult result;
    result.num_points = grid.size();
    result.times.reserve(steps / save_every + 2);
    result.norms.reserve(steps / save_every + 2);
    result.wavefunctions.reserve((steps / save_every + 2) * grid.size());

    auto save_frame = [&](std::size_t step) {
        result.times.push_back(static_cast<double>(step) * timestep);
        result.norms.push_back(continuous_norm(wavefunction, hamiltonian.spacing));
        result.wavefunctions.emplace_back(0.0, 0.0);
        result.wavefunctions.insert(
            result.wavefunctions.end(), wavefunction.begin(), wavefunction.end()
        );
        result.wavefunctions.emplace_back(0.0, 0.0);
    };
    save_frame(0);

    std::vector<std::complex<double>> right_hand_side(interior_size);
    for (std::size_t step = 1; step <= steps; ++step) {
        for (std::size_t i = 0; i < interior_size; ++i) {
            right_hand_side[i] =
                (1.0 - 0.5 * imaginary_unit * timestep * hamiltonian.diagonal[i])
                * wavefunction[i];
            if (i > 0) {
                right_hand_side[i] -=
                    0.5 * imaginary_unit * timestep
                    * hamiltonian.off_diagonal[i - 1] * wavefunction[i - 1];
            }
            if (i + 1 < interior_size) {
                right_hand_side[i] -=
                    0.5 * imaginary_unit * timestep
                    * hamiltonian.off_diagonal[i] * wavefunction[i + 1];
            }
        }
        wavefunction =
            solve_complex_tridiagonal(left_diagonal, left_off, right_hand_side);
        if (step % save_every == 0 || step == steps) {
            save_frame(step);
        }
    }
    result.num_frames = result.times.size();
    return result;
}

}  // namespace zynnova::dft
