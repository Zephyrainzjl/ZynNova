#include <zynnova/zivar/matrix_free.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace zynnova::zivar {

namespace {

void require_extent(
    const ScalarView& values,
    std::size_t expected,
    const char* argument
) {
    if (values.extent(0) != expected) {
        throw std::invalid_argument(argument);
    }
}

double minimum_value(const ScalarView& values) {
    if (values.extent(0) == 0) {
        return std::numeric_limits<double>::infinity();
    }
    double minimum = std::numeric_limits<double>::max();
    Kokkos::parallel_reduce(
        "zivar_minimum_value",
        Kokkos::RangePolicy<ExecutionSpace>(0, values.extent(0)),
        KOKKOS_LAMBDA(const std::size_t index, double& local_minimum) {
            if (values(index) < local_minimum) {
                local_minimum = values(index);
            }
        },
        Kokkos::Min<double>(minimum)
    );
    return minimum;
}

}  // namespace

SymmetricSparseOperator::SymmetricSparseOperator(
    std::size_t atom_count,
    std::size_t edge_count
)
    : atom_count_(atom_count),
      edge_count_(edge_count),
      onsite_("zivar_onsite", atom_count),
      edge_i_("zivar_edge_i", edge_count),
      edge_j_("zivar_edge_j", edge_count),
      coupling_("zivar_coupling", edge_count) {}

std::size_t SymmetricSparseOperator::atom_count() const noexcept {
    return atom_count_;
}

std::size_t SymmetricSparseOperator::edge_count() const noexcept {
    return edge_count_;
}

ScalarView& SymmetricSparseOperator::onsite() noexcept { return onsite_; }
const ScalarView& SymmetricSparseOperator::onsite() const noexcept { return onsite_; }
IndexView& SymmetricSparseOperator::edge_i() noexcept { return edge_i_; }
const IndexView& SymmetricSparseOperator::edge_i() const noexcept { return edge_i_; }
IndexView& SymmetricSparseOperator::edge_j() noexcept { return edge_j_; }
const IndexView& SymmetricSparseOperator::edge_j() const noexcept { return edge_j_; }
ScalarView& SymmetricSparseOperator::coupling() noexcept { return coupling_; }
const ScalarView& SymmetricSparseOperator::coupling() const noexcept { return coupling_; }

void apply_operator(
    const SymmetricSparseOperator& matrix,
    const ScalarView& x,
    const ScalarView& y
) {
    require_extent(x, matrix.atom_count(), "x has the wrong extent");
    require_extent(y, matrix.atom_count(), "y has the wrong extent");

    const auto onsite = matrix.onsite();
    const auto atom_count = matrix.atom_count();
    Kokkos::parallel_for(
        "zivar_onsite_matvec",
        Kokkos::RangePolicy<ExecutionSpace>(0, atom_count),
        KOKKOS_LAMBDA(const std::size_t atom) {
            y(atom) = onsite(atom) * x(atom);
        }
    );

    const auto edge_i = matrix.edge_i();
    const auto edge_j = matrix.edge_j();
    const auto coupling = matrix.coupling();
    const auto edge_count = matrix.edge_count();
    Kokkos::parallel_for(
        "zivar_symmetric_edge_matvec",
        Kokkos::RangePolicy<ExecutionSpace>(0, edge_count),
        KOKKOS_LAMBDA(const std::size_t edge) {
            const auto first = edge_i(edge);
            const auto second = edge_j(edge);
            const double weight = coupling(edge);
            Kokkos::atomic_add(&y(first), weight * x(second));
            Kokkos::atomic_add(&y(second), weight * x(first));
        }
    );
    Kokkos::fence();
}

double dot(const ScalarView& left, const ScalarView& right) {
    require_extent(right, left.extent(0), "dot operands have different extents");
    double result = 0.0;
    Kokkos::parallel_reduce(
        "zivar_dot",
        Kokkos::RangePolicy<ExecutionSpace>(0, left.extent(0)),
        KOKKOS_LAMBDA(const std::size_t index, double& update) {
            update += left(index) * right(index);
        },
        result
    );
    return result;
}

double l2_norm(const ScalarView& values) {
    return std::sqrt(std::max(0.0, dot(values, values)));
}

void axpby(
    double alpha,
    const ScalarView& x,
    double beta,
    const ScalarView& y
) {
    require_extent(y, x.extent(0), "axpby operands have different extents");
    Kokkos::parallel_for(
        "zivar_axpby",
        Kokkos::RangePolicy<ExecutionSpace>(0, x.extent(0)),
        KOKKOS_LAMBDA(const std::size_t index) {
            y(index) = alpha * x(index) + beta * y(index);
        }
    );
    Kokkos::fence();
}

void apply_jacobi_preconditioner(
    const SymmetricSparseOperator& matrix,
    const ScalarView& residual,
    const ScalarView& preconditioned
) {
    require_extent(
        residual, matrix.atom_count(), "residual has the wrong extent"
    );
    require_extent(
        preconditioned,
        matrix.atom_count(),
        "preconditioned residual has the wrong extent"
    );
    const auto onsite = matrix.onsite();
    Kokkos::parallel_for(
        "zivar_jacobi_preconditioner",
        Kokkos::RangePolicy<ExecutionSpace>(0, matrix.atom_count()),
        KOKKOS_LAMBDA(const std::size_t atom) {
            preconditioned(atom) = residual(atom) / onsite(atom);
        }
    );
    Kokkos::fence();
}

double total_constraint_residual(const ScalarView& values, double target) {
    double total = 0.0;
    Kokkos::parallel_reduce(
        "zivar_total_constraint",
        Kokkos::RangePolicy<ExecutionSpace>(0, values.extent(0)),
        KOKKOS_LAMBDA(const std::size_t index, double& update) {
            update += values(index);
        },
        total
    );
    return total - target;
}

void project_to_total(const ScalarView& values, double target) {
    const auto count = values.extent(0);
    if (count == 0) {
        if (target != 0.0) {
            throw std::invalid_argument(
                "a non-zero total cannot be imposed on an empty vector"
            );
        }
        return;
    }
    const double correction =
        -total_constraint_residual(values, target) / static_cast<double>(count);
    Kokkos::parallel_for(
        "zivar_project_total_constraint",
        Kokkos::RangePolicy<ExecutionSpace>(0, count),
        KOKKOS_LAMBDA(const std::size_t index) {
            values(index) += correction;
        }
    );
    Kokkos::fence();
}

PcgReport pcg_solve(
    const SymmetricSparseOperator& matrix,
    const ScalarView& rhs,
    const ScalarView& solution,
    const PcgOptions& options
) {
    require_extent(rhs, matrix.atom_count(), "rhs has the wrong extent");
    require_extent(
        solution, matrix.atom_count(), "solution has the wrong extent"
    );
    if (options.absolute_tolerance < 0.0 || options.relative_tolerance < 0.0) {
        throw std::invalid_argument("PCG tolerances must be non-negative");
    }
    if (options.maximum_iterations < 0 || options.minimum_curvature < 0.0) {
        throw std::invalid_argument("invalid PCG iteration or curvature limit");
    }

    PcgReport report;
    report.rhs_norm = l2_norm(rhs);

    ScalarView residual("zivar_pcg_residual", matrix.atom_count());
    ScalarView preconditioned("zivar_pcg_preconditioned", matrix.atom_count());
    ScalarView direction("zivar_pcg_direction", matrix.atom_count());
    ScalarView action("zivar_pcg_action", matrix.atom_count());

    apply_operator(matrix, solution, action);
    Kokkos::parallel_for(
        "zivar_pcg_initial_residual",
        Kokkos::RangePolicy<ExecutionSpace>(0, matrix.atom_count()),
        KOKKOS_LAMBDA(const std::size_t atom) {
            residual(atom) = rhs(atom) - action(atom);
        }
    );
    Kokkos::fence();

    report.initial_residual_norm = l2_norm(residual);
    report.residual_norm = report.initial_residual_norm;
    const double tolerance = options.absolute_tolerance
        + options.relative_tolerance * report.rhs_norm;
    if (!std::isfinite(report.residual_norm)) {
        report.termination = PcgTermination::non_finite_residual;
        return report;
    }
    if (report.residual_norm <= tolerance) {
        report.termination = PcgTermination::converged;
        report.converged = true;
        return report;
    }

    if (
        options.use_jacobi_preconditioner
        && (!(minimum_value(matrix.onsite()) > 0.0))
    ) {
        report.termination = PcgTermination::invalid_preconditioner;
        return report;
    }
    if (options.use_jacobi_preconditioner) {
        apply_jacobi_preconditioner(matrix, residual, preconditioned);
    } else {
        Kokkos::deep_copy(preconditioned, residual);
    }
    Kokkos::deep_copy(direction, preconditioned);
    double residual_dot_preconditioned = dot(residual, preconditioned);

    for (int iteration = 0; iteration < options.maximum_iterations; ++iteration) {
        apply_operator(matrix, direction, action);
        const double curvature = dot(direction, action);
        if (
            !std::isfinite(curvature)
            || curvature <= options.minimum_curvature
        ) {
            report.termination = PcgTermination::non_positive_curvature;
            report.iterations = iteration;
            return report;
        }

        const double alpha = residual_dot_preconditioned / curvature;
        axpby(alpha, direction, 1.0, solution);
        axpby(-alpha, action, 1.0, residual);
        report.iterations = iteration + 1;
        report.residual_norm = l2_norm(residual);
        if (!std::isfinite(report.residual_norm)) {
            report.termination = PcgTermination::non_finite_residual;
            return report;
        }
        if (report.residual_norm <= tolerance) {
            report.termination = PcgTermination::converged;
            report.converged = true;
            return report;
        }

        if (options.use_jacobi_preconditioner) {
            apply_jacobi_preconditioner(matrix, residual, preconditioned);
        } else {
            Kokkos::deep_copy(preconditioned, residual);
        }
        const double next_residual_dot_preconditioned =
            dot(residual, preconditioned);
        const double beta =
            next_residual_dot_preconditioned / residual_dot_preconditioned;
        axpby(1.0, preconditioned, beta, direction);
        residual_dot_preconditioned = next_residual_dot_preconditioned;
    }

    report.termination = PcgTermination::maximum_iterations;
    return report;
}

const char* termination_name(PcgTermination termination) noexcept {
    switch (termination) {
        case PcgTermination::converged:
            return "converged";
        case PcgTermination::maximum_iterations:
            return "maximum_iterations";
        case PcgTermination::non_positive_curvature:
            return "non_positive_curvature";
        case PcgTermination::invalid_preconditioner:
            return "invalid_preconditioner";
        case PcgTermination::non_finite_residual:
            return "non_finite_residual";
    }
    return "unknown";
}

}  // namespace zynnova::zivar
