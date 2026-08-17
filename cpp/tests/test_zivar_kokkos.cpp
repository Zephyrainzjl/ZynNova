#include <zynnova/zivar/matrix_free.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using zynnova::zivar::ScalarView;
using zynnova::zivar::SymmetricSparseOperator;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

SymmetricSparseOperator make_operator() {
    SymmetricSparseOperator matrix(4, 4);
    auto onsite = Kokkos::create_mirror_view(matrix.onsite());
    auto edge_i = Kokkos::create_mirror_view(matrix.edge_i());
    auto edge_j = Kokkos::create_mirror_view(matrix.edge_j());
    auto coupling = Kokkos::create_mirror_view(matrix.coupling());

    const std::array<double, 4> diagonal{4.0, 5.0, 6.0, 7.0};
    const std::array<int, 4> first{0, 1, 2, 0};
    const std::array<int, 4> second{1, 2, 3, 3};
    const std::array<double, 4> weights{-0.5, 0.25, -1.0, 0.1};
    for (std::size_t index = 0; index < diagonal.size(); ++index) {
        onsite(index) = diagonal[index];
        edge_i(index) = first[index];
        edge_j(index) = second[index];
        coupling(index) = weights[index];
    }
    Kokkos::deep_copy(matrix.onsite(), onsite);
    Kokkos::deep_copy(matrix.edge_i(), edge_i);
    Kokkos::deep_copy(matrix.edge_j(), edge_j);
    Kokkos::deep_copy(matrix.coupling(), coupling);
    return matrix;
}

ScalarView device_vector(
    const std::string& label,
    const std::array<double, 4>& values
) {
    ScalarView result(label, values.size());
    auto host = Kokkos::create_mirror_view(result);
    for (std::size_t index = 0; index < values.size(); ++index) {
        host(index) = values[index];
    }
    Kokkos::deep_copy(result, host);
    return result;
}

void test_matrix_free_matvec() {
    const auto matrix = make_operator();
    const std::array<double, 4> exact{1.0, -2.0, 0.5, 3.0};
    const auto x = device_vector("test_x", exact);
    ScalarView action("test_action", exact.size());
    zynnova::zivar::apply_operator(matrix, x, action);
    const auto host_action = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), action
    );

    const std::array<double, 4> expected{
        4.0 * exact[0] - 0.5 * exact[1] + 0.1 * exact[3],
        5.0 * exact[1] - 0.5 * exact[0] + 0.25 * exact[2],
        6.0 * exact[2] + 0.25 * exact[1] - exact[3],
        7.0 * exact[3] - exact[2] + 0.1 * exact[0],
    };
    for (std::size_t atom = 0; atom < expected.size(); ++atom) {
        require(
            std::abs(host_action(atom) - expected[atom]) < 1.0e-13,
            "matrix-free symmetric matvec failed"
        );
    }
}

void test_pcg_and_constraint_primitives() {
    const auto matrix = make_operator();
    const std::array<double, 4> exact{1.0, -2.0, 0.5, 3.0};
    const auto exact_device = device_vector("exact_solution", exact);
    ScalarView rhs("test_rhs", exact.size());
    zynnova::zivar::apply_operator(matrix, exact_device, rhs);

    ScalarView solution("test_solution", exact.size());
    Kokkos::deep_copy(solution, 0.0);
    zynnova::zivar::PcgOptions options;
    options.absolute_tolerance = 1.0e-13;
    options.relative_tolerance = 1.0e-13;
    options.maximum_iterations = 20;
    const auto report = zynnova::zivar::pcg_solve(
        matrix, rhs, solution, options
    );
    require(report.converged, "PCG failed to converge on an SPD operator");
    require(report.iterations <= 8, "PCG convergence regressed");

    const auto host_solution = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), solution
    );
    for (std::size_t atom = 0; atom < exact.size(); ++atom) {
        require(
            std::abs(host_solution(atom) - exact[atom]) < 2.0e-12,
            "PCG solution does not match the reference"
        );
    }

    constexpr double target = 2.75;
    zynnova::zivar::project_to_total(solution, target);
    require(
        std::abs(zynnova::zivar::total_constraint_residual(solution, target))
            < 1.0e-13,
        "total constraint projection failed"
    );
}

}  // namespace

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    int status = EXIT_SUCCESS;
    {
        try {
            test_matrix_free_matvec();
            test_pcg_and_constraint_primitives();
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            status = EXIT_FAILURE;
        }
    }
    Kokkos::finalize();
    if (status == EXIT_SUCCESS) {
        std::cout << "Ziver Kokkos numerical-core tests passed\n";
    }
    return status;
}
