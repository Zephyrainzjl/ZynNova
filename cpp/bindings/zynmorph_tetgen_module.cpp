#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tetgen.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

struct RefinementZone {
  std::array<double, 3> center{};
  double radius_squared{0.0};
  double maximum_volume{0.0};
};

thread_local const std::vector<RefinementZone>* active_zones = nullptr;
std::mutex tetgen_mutex;

bool local_tet_unsuitable(
    REAL* pa,
    REAL* pb,
    REAL* pc,
    REAL* pd,
    REAL* circumcenter,
    REAL volume) {
  if (active_zones == nullptr || active_zones->empty()) {
    return false;
  }
  std::array<double, 3> center{};
  if (circumcenter != nullptr) {
    center = {
        static_cast<double>(circumcenter[0]),
        static_cast<double>(circumcenter[1]),
        static_cast<double>(circumcenter[2]),
    };
  } else {
    for (int axis = 0; axis < 3; ++axis) {
      center[static_cast<std::size_t>(axis)] =
          0.25 * (static_cast<double>(pa[axis]) + static_cast<double>(pb[axis]) +
                  static_cast<double>(pc[axis]) + static_cast<double>(pd[axis]));
    }
  }
  const double absolute_volume = std::abs(static_cast<double>(volume));
  for (const auto& zone : *active_zones) {
    const double dx = center[0] - zone.center[0];
    const double dy = center[1] - zone.center[1];
    const double dz = center[2] - zone.center[2];
    if (dx * dx + dy * dy + dz * dz <= zone.radius_squared &&
        absolute_volume > zone.maximum_volume) {
      return true;
    }
  }
  return false;
}

std::string tetgen_error_message(const int code) {
  switch (code) {
    case 1:
      return "TetGen failed: out of memory";
    case 2:
      return "TetGen failed: internal bug";
    case 3:
      return "TetGen failed: self-intersection or invalid PLC";
    case 4:
      return "TetGen failed: very small input feature";
    case 5:
      return "TetGen failed: two very close input facets";
    case 10:
      return "TetGen failed: input error";
    default:
      return "TetGen failed with error code " + std::to_string(code);
  }
}

template <typename T>
void require_matrix_shape(
    const py::buffer_info& info,
    const py::ssize_t width,
    const char* name) {
  (void)sizeof(T);
  if (info.ndim != 2 || info.shape[1] != width) {
    std::ostringstream stream;
    stream << name << " must have shape (N, " << width << ")";
    throw std::invalid_argument(stream.str());
  }
}

py::dict tetrahedralize_plc(
    py::array_t<double, py::array::c_style | py::array::forcecast> points,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> facets,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> facet_markers,
    py::array_t<double, py::array::c_style | py::array::forcecast> region_seeds,
    py::array_t<double, py::array::c_style | py::array::forcecast> holes,
    py::array_t<double, py::array::c_style | py::array::forcecast> facet_constraints,
    py::array_t<double, py::array::c_style | py::array::forcecast> local_zones,
    const double radius_edge_ratio,
    const double minimum_dihedral_degrees,
    const int optimization_level,
    const int maximum_steiner_points,
    const bool consistency_check,
    const bool conforming_delaunay,
    const bool quiet) {
  const auto point_info = points.request();
  const auto facet_info = facets.request();
  const auto marker_info = facet_markers.request();
  const auto region_info = region_seeds.request();
  const auto hole_info = holes.request();
  const auto constraint_info = facet_constraints.request();
  const auto zone_info = local_zones.request();

  require_matrix_shape<double>(point_info, 3, "points");
  require_matrix_shape<std::int64_t>(facet_info, 3, "facets");
  require_matrix_shape<double>(region_info, 5, "region_seeds");
  require_matrix_shape<double>(hole_info, 3, "holes");
  require_matrix_shape<double>(constraint_info, 2, "facet_constraints");
  require_matrix_shape<double>(zone_info, 5, "local_zones");
  if (marker_info.ndim != 1 || marker_info.shape[0] != facet_info.shape[0]) {
    throw std::invalid_argument("facet_markers must have one entry per facet");
  }
  if (point_info.shape[0] < 4 || facet_info.shape[0] < 4) {
    throw std::invalid_argument("TetGen PLC requires at least four points and four facets");
  }
  if (point_info.shape[0] > std::numeric_limits<int>::max() ||
      facet_info.shape[0] > std::numeric_limits<int>::max() ||
      region_info.shape[0] > std::numeric_limits<int>::max()) {
    throw std::overflow_error("TetGen input exceeds 32-bit indexing limits");
  }
  if (!(std::isfinite(radius_edge_ratio) && radius_edge_ratio > 1.0)) {
    throw std::invalid_argument("radius_edge_ratio must be finite and greater than one");
  }
  if (!(std::isfinite(minimum_dihedral_degrees) &&
        minimum_dihedral_degrees >= 0.0 && minimum_dihedral_degrees < 60.0)) {
    throw std::invalid_argument("minimum_dihedral_degrees must lie in [0, 60)");
  }
  if (optimization_level < 0 || optimization_level > 10) {
    throw std::invalid_argument("optimization_level must lie in [0, 10]");
  }
  if (maximum_steiner_points < -1) {
    throw std::invalid_argument("maximum_steiner_points must be -1 or non-negative");
  }

  tetgenio input;
  tetgenio output;
  input.firstnumber = 0;

  input.numberofpoints = static_cast<int>(point_info.shape[0]);
  input.pointlist = new REAL[static_cast<std::size_t>(input.numberofpoints) * 3];
  const auto* point_data = static_cast<const double*>(point_info.ptr);
  for (std::size_t index = 0;
       index < static_cast<std::size_t>(input.numberofpoints) * 3;
       ++index) {
    const double value = point_data[index];
    if (!std::isfinite(value)) {
      throw std::invalid_argument("points contain non-finite coordinates");
    }
    input.pointlist[index] = static_cast<REAL>(value);
  }

  input.numberoffacets = static_cast<int>(facet_info.shape[0]);
  input.facetlist = new tetgenio::facet[static_cast<std::size_t>(input.numberoffacets)];
  input.facetmarkerlist = new int[static_cast<std::size_t>(input.numberoffacets)];
  const auto* facet_data = static_cast<const std::int64_t*>(facet_info.ptr);
  const auto* marker_data = static_cast<const std::int32_t*>(marker_info.ptr);
  for (int face_index = 0; face_index < input.numberoffacets; ++face_index) {
    auto& facet = input.facetlist[face_index];
    facet.numberofpolygons = 1;
    facet.polygonlist = new tetgenio::polygon[1];
    facet.numberofholes = 0;
    facet.holelist = nullptr;
    auto& polygon = facet.polygonlist[0];
    polygon.numberofvertices = 3;
    polygon.vertexlist = new int[3];
    for (int corner = 0; corner < 3; ++corner) {
      const auto value = facet_data[static_cast<std::size_t>(face_index) * 3 + corner];
      if (value < 0 || value >= input.numberofpoints ||
          value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("facet contains an out-of-range point index");
      }
      polygon.vertexlist[corner] = static_cast<int>(value);
    }
    input.facetmarkerlist[face_index] = static_cast<int>(marker_data[face_index]);
  }

  input.numberofregions = static_cast<int>(region_info.shape[0]);
  if (input.numberofregions <= 0) {
    throw std::invalid_argument("at least one TetGen region seed is required");
  }
  input.regionlist = new REAL[static_cast<std::size_t>(input.numberofregions) * 5];
  const auto* region_data = static_cast<const double*>(region_info.ptr);
  for (std::size_t index = 0;
       index < static_cast<std::size_t>(input.numberofregions) * 5;
       ++index) {
    const double value = region_data[index];
    if (!std::isfinite(value)) {
      throw std::invalid_argument("region_seeds contain non-finite values");
    }
    input.regionlist[index] = static_cast<REAL>(value);
  }

  input.numberofholes = static_cast<int>(hole_info.shape[0]);
  if (input.numberofholes > 0) {
    input.holelist = new REAL[static_cast<std::size_t>(input.numberofholes) * 3];
    const auto* hole_data = static_cast<const double*>(hole_info.ptr);
    for (std::size_t index = 0;
         index < static_cast<std::size_t>(input.numberofholes) * 3;
         ++index) {
      const double value = hole_data[index];
      if (!std::isfinite(value)) {
        throw std::invalid_argument("holes contain non-finite coordinates");
      }
      input.holelist[index] = static_cast<REAL>(value);
    }
  }

  input.numberoffacetconstraints = static_cast<int>(constraint_info.shape[0]);
  if (input.numberoffacetconstraints > 0) {
    input.facetconstraintlist =
        new REAL[static_cast<std::size_t>(input.numberoffacetconstraints) * 2];
    const auto* constraint_data = static_cast<const double*>(constraint_info.ptr);
    for (int index = 0; index < input.numberoffacetconstraints; ++index) {
      const double marker = constraint_data[static_cast<std::size_t>(index) * 2];
      const double area = constraint_data[static_cast<std::size_t>(index) * 2 + 1];
      if (!std::isfinite(marker) || !std::isfinite(area) || area <= 0.0) {
        throw std::invalid_argument("facet constraints require finite marker/positive area");
      }
      input.facetconstraintlist[static_cast<std::size_t>(index) * 2] =
          static_cast<REAL>(marker);
      input.facetconstraintlist[static_cast<std::size_t>(index) * 2 + 1] =
          static_cast<REAL>(area);
    }
  }

  std::vector<RefinementZone> zones;
  zones.reserve(static_cast<std::size_t>(zone_info.shape[0]));
  const auto* zone_data = static_cast<const double*>(zone_info.ptr);
  for (py::ssize_t index = 0; index < zone_info.shape[0]; ++index) {
    const auto offset = static_cast<std::size_t>(index) * 5;
    const double radius = zone_data[offset + 3];
    const double maximum_volume = zone_data[offset + 4];
    if (!(std::isfinite(radius) && radius > 0.0 &&
          std::isfinite(maximum_volume) && maximum_volume > 0.0)) {
      throw std::invalid_argument("local zones require positive radius and maximum volume");
    }
    RefinementZone zone;
    for (int axis = 0; axis < 3; ++axis) {
      const double value = zone_data[offset + static_cast<std::size_t>(axis)];
      if (!std::isfinite(value)) {
        throw std::invalid_argument("local zone center contains a non-finite value");
      }
      zone.center[static_cast<std::size_t>(axis)] = value;
    }
    zone.radius_squared = radius * radius;
    zone.maximum_volume = maximum_volume;
    zones.push_back(zone);
  }
  if (!zones.empty()) {
    input.tetunsuitable = &local_tet_unsuitable;
  }

  std::ostringstream switches;
  switches.precision(17);
  switches << "pzAafq" << radius_edge_ratio << "/" << minimum_dihedral_degrees;
  switches << "O" << optimization_level;
  if (conforming_delaunay) {
    switches << "D";
  }
  if (consistency_check) {
    switches << "C";
  }
  if (quiet) {
    switches << "Q";
  }
  if (maximum_steiner_points >= 0) {
    switches << "S" << maximum_steiner_points;
  }
  if (!zones.empty()) {
    switches << "u";
  }
  const std::string switch_string = switches.str();
  std::vector<char> mutable_switches(switch_string.begin(), switch_string.end());
  mutable_switches.push_back('\0');
  tetgenbehavior behavior;
  if (!behavior.parse_commandline(mutable_switches.data())) {
    throw std::invalid_argument("TetGen rejected switches: " + switch_string);
  }

  {
    std::lock_guard<std::mutex> lock(tetgen_mutex);
    active_zones = &zones;
    try {
      py::gil_scoped_release release;
      tetrahedralize(&behavior, &input, &output, nullptr, nullptr);
    } catch (const int error_code) {
      active_zones = nullptr;
      throw std::runtime_error(tetgen_error_message(error_code));
    } catch (...) {
      active_zones = nullptr;
      throw;
    }
    active_zones = nullptr;
  }

  if (output.numberofcorners != 4) {
    throw std::runtime_error("TetGen returned a non-Tet4 mesh unexpectedly");
  }
  if (output.numberofpoints <= 0 || output.numberoftetrahedra <= 0 ||
      output.pointlist == nullptr || output.tetrahedronlist == nullptr) {
    throw std::runtime_error("TetGen returned an empty volume mesh");
  }

  py::array_t<double> output_points(
      {static_cast<py::ssize_t>(output.numberofpoints), py::ssize_t{3}});
  auto* output_point_data = output_points.mutable_data();
  for (std::size_t index = 0;
       index < static_cast<std::size_t>(output.numberofpoints) * 3;
       ++index) {
    output_point_data[index] = static_cast<double>(output.pointlist[index]);
  }

  py::array_t<std::int64_t> output_tets(
      {static_cast<py::ssize_t>(output.numberoftetrahedra), py::ssize_t{4}});
  auto* output_tet_data = output_tets.mutable_data();
  for (std::size_t index = 0;
       index < static_cast<std::size_t>(output.numberoftetrahedra) * 4;
       ++index) {
    output_tet_data[index] = static_cast<std::int64_t>(output.tetrahedronlist[index]);
  }

  py::array_t<double> output_attributes(
      {static_cast<py::ssize_t>(output.numberoftetrahedra)});
  auto* output_attribute_data = output_attributes.mutable_data();
  if (output.numberoftetrahedronattributes < 1 ||
      output.tetrahedronattributelist == nullptr) {
    throw std::runtime_error("TetGen returned no region attributes; -A contract failed");
  }
  for (int index = 0; index < output.numberoftetrahedra; ++index) {
    output_attribute_data[index] = static_cast<double>(
        output.tetrahedronattributelist[
            static_cast<std::size_t>(index) * output.numberoftetrahedronattributes]);
  }

  py::array_t<std::int64_t> output_faces(
      {static_cast<py::ssize_t>(std::max(output.numberoftrifaces, 0)), py::ssize_t{3}});
  py::array_t<std::int32_t> output_face_markers(
      {static_cast<py::ssize_t>(std::max(output.numberoftrifaces, 0))});
  if (output.numberoftrifaces > 0 && output.trifacelist != nullptr) {
    auto* output_face_data = output_faces.mutable_data();
    auto* output_face_marker_data = output_face_markers.mutable_data();
    for (int face_index = 0; face_index < output.numberoftrifaces; ++face_index) {
      for (int corner = 0; corner < 3; ++corner) {
        output_face_data[static_cast<std::size_t>(face_index) * 3 + corner] =
            static_cast<std::int64_t>(
                output.trifacelist[static_cast<std::size_t>(face_index) * 3 + corner]);
      }
      output_face_marker_data[face_index] =
          output.trifacemarkerlist == nullptr
              ? 0
              : static_cast<std::int32_t>(output.trifacemarkerlist[face_index]);
    }
  }

  py::dict result;
  result["points"] = std::move(output_points);
  result["tetrahedra"] = std::move(output_tets);
  result["region_attributes"] = std::move(output_attributes);
  result["trifaces"] = std::move(output_faces);
  result["triface_markers"] = std::move(output_face_markers);
  result["switches"] = switch_string;
  result["version"] = "TetGen 1.6.0";
  return result;
}

}  // namespace

PYBIND11_MODULE(_zynmorph_tetgen_native, module) {
  module.doc() =
      "ZynMorph pybind11 interface to the vendored TetGen 1.6.0 C++ kernel";
  module.attr("tetgen_version") = "TetGen 1.6.0";
  module.attr("tetgen_license") = "AGPL-3.0-or-later";
  module.def(
      "tetrahedralize",
      &tetrahedralize_plc,
      py::arg("points"),
      py::arg("facets"),
      py::arg("facet_markers"),
      py::arg("region_seeds"),
      py::arg("holes"),
      py::arg("facet_constraints"),
      py::arg("local_zones"),
      py::arg("radius_edge_ratio") = 1.45,
      py::arg("minimum_dihedral_degrees") = 8.0,
      py::arg("optimization_level") = 2,
      py::arg("maximum_steiner_points") = -1,
      py::arg("consistency_check") = true,
      py::arg("conforming_delaunay") = true,
      py::arg("quiet") = true,
      R"doc(
Generate a region-partitioned adaptive Tet4 mesh from a triangular PLC.

TetGen source and the linked native component are AGPL-3.0-or-later.  Region
seeds use the TetGen five-value layout ``x,y,z,attribute,max_volume``.  Local
spherical refinement zones activate TetGen's ``-u`` sizing callback.
)doc");
}
