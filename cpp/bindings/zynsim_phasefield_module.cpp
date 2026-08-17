#include "zynnova/zynsim/phasefield.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace py = pybind11;
namespace pf = zynnova::zynsim::phasefield;

namespace {

pf::CartesianGrid make_grid(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& field,
    const std::vector<double>& spacing,
    const std::vector<std::string>& boundary) {
  pf::CartesianGrid grid;
  grid.shape.reserve(static_cast<std::size_t>(field.ndim()));
  for (py::ssize_t axis = 0; axis < field.ndim(); ++axis) {
    grid.shape.push_back(static_cast<std::size_t>(field.shape(axis)));
  }
  grid.spacing = spacing;
  grid.boundary.reserve(boundary.size());
  for (const auto& item : boundary) {
    grid.boundary.push_back(pf::parse_boundary(item));
  }
  grid.validate();
  return grid;
}

std::vector<double> to_vector(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& field) {
  const auto* begin = field.data();
  return std::vector<double>(begin, begin + field.size());
}

py::array_t<double> to_array(
    const std::vector<double>& values,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& reference) {
  std::vector<py::ssize_t> shape;
  shape.reserve(static_cast<std::size_t>(reference.ndim()));
  for (py::ssize_t axis = 0; axis < reference.ndim(); ++axis) {
    shape.push_back(reference.shape(axis));
  }
  py::array_t<double> output(shape);
  std::memcpy(output.mutable_data(), values.data(), values.size() * sizeof(double));
  return output;
}

}  // namespace

PYBIND11_MODULE(_zynsim_phasefield_native, module) {
  module.doc() = "ZynSim 1D/2D/3D phase-field finite-difference kernels";

  module.def(
      "laplacian",
      [](const py::array_t<double, py::array::c_style | py::array::forcecast>& field,
         const std::vector<double>& spacing,
         const std::vector<std::string>& boundary,
         int order) {
        const auto grid = make_grid(field, spacing, boundary);
        return to_array(pf::laplacian(to_vector(field), grid, order), field);
      },
      py::arg("field"),
      py::arg("spacing"),
      py::arg("boundary"),
      py::arg("order") = 4);

  module.def(
      "biharmonic",
      [](const py::array_t<double, py::array::c_style | py::array::forcecast>& field,
         const std::vector<double>& spacing,
         const std::vector<std::string>& boundary,
         int order) {
        const auto grid = make_grid(field, spacing, boundary);
        return to_array(pf::biharmonic(to_vector(field), grid, order), field);
      },
      py::arg("field"),
      py::arg("spacing"),
      py::arg("boundary"),
      py::arg("order") = 4);

  module.def(
      "gradient_energy",
      [](const py::array_t<double, py::array::c_style | py::array::forcecast>& field,
         const std::vector<double>& spacing,
         const std::vector<std::string>& boundary,
         int order) {
        const auto grid = make_grid(field, spacing, boundary);
        return pf::gradient_energy(to_vector(field), grid, order);
      },
      py::arg("field"),
      py::arg("spacing"),
      py::arg("boundary"),
      py::arg("order") = 4);

  module.def(
      "integral",
      [](const py::array_t<double, py::array::c_style | py::array::forcecast>& field,
         const std::vector<double>& spacing,
         const std::vector<std::string>& boundary) {
        const auto grid = make_grid(field, spacing, boundary);
        return pf::integral(to_vector(field), grid);
      },
      py::arg("field"),
      py::arg("spacing"),
      py::arg("boundary"));

#ifdef _OPENMP
  module.attr("openmp_enabled") = true;
#else
  module.attr("openmp_enabled") = false;
#endif
}
