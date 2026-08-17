#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace zynnova::zynsim::phasefield {

enum class BoundaryCondition {
  Periodic,
  Neumann,
  Dirichlet,
};

struct CartesianGrid {
  std::vector<std::size_t> shape;
  std::vector<double> spacing;
  std::vector<BoundaryCondition> boundary;

  [[nodiscard]] std::size_t dimensions() const noexcept { return shape.size(); }
  [[nodiscard]] std::size_t size() const noexcept;
  void validate() const;
};

[[nodiscard]] BoundaryCondition parse_boundary(const std::string& name);

[[nodiscard]] std::vector<double> laplacian(
    const std::vector<double>& field,
    const CartesianGrid& grid,
    int order = 4);

[[nodiscard]] std::vector<double> biharmonic(
    const std::vector<double>& field,
    const CartesianGrid& grid,
    int order = 4);

[[nodiscard]] double gradient_energy(
    const std::vector<double>& field,
    const CartesianGrid& grid,
    int order = 4);

[[nodiscard]] double integral(
    const std::vector<double>& field,
    const CartesianGrid& grid);

}  // namespace zynnova::zynsim::phasefield
