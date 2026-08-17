#include "zynnova/zynsim/voxel.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using namespace zynnova::zynsim;

PYBIND11_MODULE(_zynsim_voxel_native, module) {
  module.doc() = "OpenMP kernels for ZynSim image/voxel reconstruction and meshing";

  module.def(
      "threshold_labels_u8",
      [](py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> values,
         const std::vector<std::uint8_t>& lower,
         const std::vector<std::uint8_t>& upper,
         const std::vector<std::int32_t>& labels,
         const std::int32_t unclassified) {
        const auto info = values.request();
        std::vector<std::int32_t> output;
        {
          py::gil_scoped_release release;
          output = threshold_labels_u8(
              static_cast<const std::uint8_t*>(info.ptr),
              static_cast<std::size_t>(info.size), lower, upper, labels, unclassified);
        }
        py::array_t<std::int32_t> result(info.shape);
        std::memcpy(result.mutable_data(), output.data(), output.size() * sizeof(std::int32_t));
        return result;
      },
      py::arg("values"), py::arg("lower"), py::arg("upper"), py::arg("labels"),
      py::arg("unclassified") = 0);

  module.def(
      "fuse_orthogonal_labels",
      [](py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> xy,
         py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> xz,
         py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> yz,
         const std::vector<std::int32_t>& phases,
         const std::vector<double>& log_priors,
         const double projection_weight) {
        const auto a = xy.request();
        const auto b = xz.request();
        const auto c = yz.request();
        if (a.ndim != 2 || b.ndim != 2 || c.ndim != 2 ||
            a.shape[0] != b.shape[0] || a.shape[1] != c.shape[0] ||
            b.shape[1] != c.shape[1]) {
          throw std::invalid_argument("xy/xz/yz shapes are inconsistent");
        }
        const auto nx = static_cast<std::size_t>(a.shape[0]);
        const auto ny = static_cast<std::size_t>(a.shape[1]);
        const auto nz = static_cast<std::size_t>(b.shape[1]);
        std::vector<std::int32_t> output;
        {
          py::gil_scoped_release release;
          output = fuse_orthogonal_labels(
              static_cast<const std::int32_t*>(a.ptr),
              static_cast<const std::int32_t*>(b.ptr),
              static_cast<const std::int32_t*>(c.ptr), nx, ny, nz,
              phases, log_priors, projection_weight);
        }
        py::array_t<std::int32_t> result({
            static_cast<py::ssize_t>(nx), static_cast<py::ssize_t>(ny),
            static_cast<py::ssize_t>(nz)});
        std::memcpy(result.mutable_data(), output.data(), output.size() * sizeof(std::int32_t));
        return result;
      },
      py::arg("xy"), py::arg("xz"), py::arg("yz"), py::arg("phases"),
      py::arg("log_priors"), py::arg("projection_weight") = 4.0);

  module.def(
      "phase_counts",
      [](py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> labels) {
        const auto info = labels.request();
        std::map<std::int32_t, std::uint64_t> counts;
        {
          py::gil_scoped_release release;
          counts = phase_counts(static_cast<const std::int32_t*>(info.ptr),
                                static_cast<std::size_t>(info.size));
        }
        return counts;
      });

  module.def(
      "interface_counts",
      [](py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> labels) {
        const auto info = labels.request();
        if (info.ndim != 3) {
          throw std::invalid_argument("labels must be three-dimensional");
        }
        std::array<std::uint64_t, 3> counts{};
        {
          py::gil_scoped_release release;
          counts = interface_counts(
              static_cast<const std::int32_t*>(info.ptr),
              static_cast<std::size_t>(info.shape[0]),
              static_cast<std::size_t>(info.shape[1]),
              static_cast<std::size_t>(info.shape[2]));
        }
        return counts;
      });

  module.def(
      "hex_connectivity_range",
      [](const std::array<std::size_t, 3>& shape,
         const std::uint64_t start, const std::uint64_t stop) {
        std::vector<std::int64_t> output;
        {
          py::gil_scoped_release release;
          output = hex_connectivity_range(shape[0], shape[1], shape[2], start, stop);
        }
        const auto rows = static_cast<py::ssize_t>(stop - start);
        py::array_t<std::int64_t> result({rows, py::ssize_t{8}});
        std::memcpy(result.mutable_data(), output.data(), output.size() * sizeof(std::int64_t));
        return result;
      },
      py::arg("shape"), py::arg("start"), py::arg("stop"));
}
