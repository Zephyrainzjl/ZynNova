#include "zynnova/zynsim/phasefield.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace zynnova::zynsim::phasefield {
namespace {

[[nodiscard]] std::vector<std::size_t> strides(const CartesianGrid& grid) {
  std::vector<std::size_t> result(grid.dimensions(), 1);
  for (std::size_t axis = grid.dimensions(); axis-- > 1;) {
    result[axis - 1] = result[axis] * grid.shape[axis];
  }
  return result;
}

[[nodiscard]] long neighbor_index(
    std::size_t flat,
    std::size_t axis,
    int offset,
    const CartesianGrid& grid,
    const std::vector<std::size_t>& grid_strides) {
  const auto coordinate = static_cast<long>((flat / grid_strides[axis]) % grid.shape[axis]);
  long shifted = coordinate + static_cast<long>(offset);
  const auto extent = static_cast<long>(grid.shape[axis]);

  switch (grid.boundary[axis]) {
    case BoundaryCondition::Periodic:
      shifted %= extent;
      if (shifted < 0) {
        shifted += extent;
      }
      break;
    case BoundaryCondition::Neumann:
      shifted = std::clamp(shifted, 0L, extent - 1L);
      break;
    case BoundaryCondition::Dirichlet:
      if (shifted < 0 || shifted >= extent) {
        return -1;
      }
      break;
  }

  const auto delta = shifted - coordinate;
  return static_cast<long>(flat) + delta * static_cast<long>(grid_strides[axis]);
}

[[nodiscard]] double sample(
    const std::vector<double>& field,
    std::size_t flat,
    std::size_t axis,
    int offset,
    const CartesianGrid& grid,
    const std::vector<std::size_t>& grid_strides) {
  const long index = neighbor_index(flat, axis, offset, grid, grid_strides);
  return index < 0 ? 0.0 : field[static_cast<std::size_t>(index)];
}

}  // namespace

std::size_t CartesianGrid::size() const noexcept {
  return std::accumulate(
      shape.begin(), shape.end(), std::size_t{1}, std::multiplies<>());
}

void CartesianGrid::validate() const {
  if (shape.empty() || shape.size() > 3) {
    throw std::invalid_argument("phase-field grids must be 1D, 2D, or 3D");
  }
  if (spacing.size() != shape.size() || boundary.size() != shape.size()) {
    throw std::invalid_argument("shape, spacing, and boundary dimensions must match");
  }
  for (std::size_t axis = 0; axis < shape.size(); ++axis) {
    if (shape[axis] < 3) {
      throw std::invalid_argument("every phase-field grid axis needs at least three points");
    }
    if (!(spacing[axis] > 0.0) || !std::isfinite(spacing[axis])) {
      throw std::invalid_argument("phase-field grid spacing must be finite and positive");
    }
  }
}

BoundaryCondition parse_boundary(const std::string& name) {
  if (name == "periodic") {
    return BoundaryCondition::Periodic;
  }
  if (name == "neumann") {
    return BoundaryCondition::Neumann;
  }
  if (name == "dirichlet") {
    return BoundaryCondition::Dirichlet;
  }
  throw std::invalid_argument("unknown phase-field boundary condition: " + name);
}

std::vector<double> laplacian(
    const std::vector<double>& field,
    const CartesianGrid& grid,
    int order) {
  grid.validate();
  if (field.size() != grid.size()) {
    throw std::invalid_argument("field size does not match phase-field grid");
  }
  if (order != 2 && order != 4) {
    throw std::invalid_argument("finite-difference order must be 2 or 4");
  }

  const auto grid_strides = strides(grid);
  std::vector<double> output(field.size(), 0.0);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t raw = 0; raw < static_cast<std::ptrdiff_t>(field.size()); ++raw) {
    const auto index = static_cast<std::size_t>(raw);
    double value = 0.0;
    for (std::size_t axis = 0; axis < grid.dimensions(); ++axis) {
      const double inverse_dx2 = 1.0 / (grid.spacing[axis] * grid.spacing[axis]);
      if (order == 2) {
        value += (
            sample(field, index, axis, 1, grid, grid_strides)
            - 2.0 * field[index]
            + sample(field, index, axis, -1, grid, grid_strides))
            * inverse_dx2;
      } else {
        value += (
            -sample(field, index, axis, 2, grid, grid_strides)
            + 16.0 * sample(field, index, axis, 1, grid, grid_strides)
            - 30.0 * field[index]
            + 16.0 * sample(field, index, axis, -1, grid, grid_strides)
            - sample(field, index, axis, -2, grid, grid_strides))
            * (inverse_dx2 / 12.0);
      }
    }
    output[index] = value;
  }
  return output;
}

std::vector<double> biharmonic(
    const std::vector<double>& field,
    const CartesianGrid& grid,
    int order) {
  return laplacian(laplacian(field, grid, order), grid, order);
}

double gradient_energy(
    const std::vector<double>& field,
    const CartesianGrid& grid,
    int order) {
  grid.validate();
  if (field.size() != grid.size()) {
    throw std::invalid_argument("field size does not match phase-field grid");
  }
  if (order != 2 && order != 4) {
    throw std::invalid_argument("finite-difference order must be 2 or 4");
  }
  const auto grid_strides = strides(grid);
  const double cell_volume = std::accumulate(
      grid.spacing.begin(), grid.spacing.end(), 1.0, std::multiplies<>());
  double total = 0.0;

#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : total) schedule(static)
#endif
  for (std::ptrdiff_t raw = 0; raw < static_cast<std::ptrdiff_t>(field.size()); ++raw) {
    const auto index = static_cast<std::size_t>(raw);
    double norm_squared = 0.0;
    for (std::size_t axis = 0; axis < grid.dimensions(); ++axis) {
      double derivative = 0.0;
      if (order == 2) {
        derivative = (
            sample(field, index, axis, -1, grid, grid_strides)
            - sample(field, index, axis, 1, grid, grid_strides))
            / (2.0 * grid.spacing[axis]);
      } else {
        derivative = (
            sample(field, index, axis, 2, grid, grid_strides)
            - 8.0 * sample(field, index, axis, 1, grid, grid_strides)
            + 8.0 * sample(field, index, axis, -1, grid, grid_strides)
            - sample(field, index, axis, -2, grid, grid_strides))
            / (12.0 * grid.spacing[axis]);
      }
      norm_squared += derivative * derivative;
    }
    total += 0.5 * norm_squared * cell_volume;
  }
  return total;
}

double integral(const std::vector<double>& field, const CartesianGrid& grid) {
  grid.validate();
  if (field.size() != grid.size()) {
    throw std::invalid_argument("field size does not match phase-field grid");
  }
  const double cell_volume = std::accumulate(
      grid.spacing.begin(), grid.spacing.end(), 1.0, std::multiplies<>());
  return std::accumulate(field.begin(), field.end(), 0.0) * cell_volume;
}

}  // namespace zynnova::zynsim::phasefield
