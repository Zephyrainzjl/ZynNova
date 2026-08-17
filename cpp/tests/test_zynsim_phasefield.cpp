#include "zynnova/zynsim/phasefield.hpp"

#include <cassert>
#include <cmath>
#include <vector>

int main() {
  namespace pf = zynnova::zynsim::phasefield;
  pf::CartesianGrid grid{
      {32},
      {2.0 * 3.14159265358979323846 / 32.0},
      {pf::BoundaryCondition::Periodic}};
  std::vector<double> field(grid.size());
  for (std::size_t index = 0; index < field.size(); ++index) {
    field[index] = std::sin(index * grid.spacing[0]);
  }
  const auto lap = pf::laplacian(field, grid, 4);
  double error = 0.0;
  for (std::size_t index = 0; index < field.size(); ++index) {
    error = std::max(error, std::abs(lap[index] + field[index]));
  }
  assert(error < 1.0e-4);
  assert(std::abs(pf::integral(field, grid)) < 1.0e-12);
  assert(pf::gradient_energy(field, grid, 4) > 0.0);
  return 0;
}
