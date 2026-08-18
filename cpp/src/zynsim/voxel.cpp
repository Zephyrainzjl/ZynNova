#include "zynnova/zynsim/voxel.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace zynnova::zynsim {

std::vector<std::int32_t> threshold_labels_u8(
    const std::uint8_t* values,
    const std::size_t count,
    const std::vector<std::uint8_t>& lower,
    const std::vector<std::uint8_t>& upper,
    const std::vector<std::int32_t>& labels,
    const std::int32_t unclassified) {
  if (values == nullptr) {
    throw std::invalid_argument("values cannot be null");
  }
  if (lower.size() != upper.size() || lower.size() != labels.size()) {
    throw std::invalid_argument("threshold arrays must share one length");
  }
  std::vector<std::int32_t> output(count, unclassified);
#pragma omp parallel for if(count > 100000)
  for (std::int64_t raw_index = 0;
       raw_index < static_cast<std::int64_t>(count); ++raw_index) {
    const auto value = values[static_cast<std::size_t>(raw_index)];
    for (std::size_t rule = 0; rule < labels.size(); ++rule) {
      if (value >= lower[rule] && value <= upper[rule]) {
        output[static_cast<std::size_t>(raw_index)] = labels[rule];
        break;
      }
    }
  }
  return output;
}

std::vector<std::int32_t> fuse_orthogonal_labels(
    const std::int32_t* xy,
    const std::int32_t* xz,
    const std::int32_t* yz,
    const std::size_t nx,
    const std::size_t ny,
    const std::size_t nz,
    const std::vector<std::int32_t>& phases,
    const std::vector<double>& log_priors,
    const double projection_weight) {
  if (xy == nullptr || xz == nullptr || yz == nullptr) {
    throw std::invalid_argument("orthogonal views cannot be null");
  }
  if (phases.empty() || phases.size() != log_priors.size()) {
    throw std::invalid_argument("phases and priors must share a non-zero length");
  }
  const std::uint64_t count = static_cast<std::uint64_t>(nx) * ny * nz;
  if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error("volume is too large for this process");
  }
  std::vector<std::int32_t> output(static_cast<std::size_t>(count));
#pragma omp parallel for collapse(2) schedule(static) if(count > 100000)
  for (std::int64_t raw_i = 0; raw_i < static_cast<std::int64_t>(nx); ++raw_i) {
    for (std::int64_t raw_j = 0; raw_j < static_cast<std::int64_t>(ny); ++raw_j) {
      const auto i = static_cast<std::size_t>(raw_i);
      const auto j = static_cast<std::size_t>(raw_j);
      for (std::size_t k = 0; k < nz; ++k) {
        const auto a = xy[i * ny + j];
        const auto b = xz[i * nz + k];
        const auto c = yz[j * nz + k];
        double best_score = -std::numeric_limits<double>::infinity();
        std::int32_t best_phase = phases.front();
        for (std::size_t p = 0; p < phases.size(); ++p) {
          const auto phase = phases[p];
          const double score = log_priors[p] + projection_weight *
              static_cast<double>((a == phase) + (b == phase) + (c == phase));
          if (score > best_score) {
            best_score = score;
            best_phase = phase;
          }
        }
        output[(i * ny + j) * nz + k] = best_phase;
      }
    }
  }
  return output;
}

std::map<std::int32_t, std::uint64_t> phase_counts(
    const std::int32_t* labels,
    const std::size_t count) {
  if (labels == nullptr) {
    throw std::invalid_argument("labels cannot be null");
  }
  std::map<std::int32_t, std::uint64_t> result;
  // A deterministic serial reduction is preferable for a small number of
  // phases; memory bandwidth dominates this pass.
  for (std::size_t index = 0; index < count; ++index) {
    ++result[labels[index]];
  }
  return result;
}

std::array<std::uint64_t, 3> interface_counts(
    const std::int32_t* labels,
    const std::size_t nx,
    const std::size_t ny,
    const std::size_t nz) {
  if (labels == nullptr) {
    throw std::invalid_argument("labels cannot be null");
  }
  std::uint64_t count_x = 0;
  std::uint64_t count_y = 0;
  std::uint64_t count_z = 0;
#pragma omp parallel for reduction(+:count_x,count_y,count_z) collapse(2) schedule(static) if(nx*ny*nz > 100000)
  for (std::int64_t raw_i = 0; raw_i < static_cast<std::int64_t>(nx); ++raw_i) {
    for (std::int64_t raw_j = 0; raw_j < static_cast<std::int64_t>(ny); ++raw_j) {
      const auto i = static_cast<std::size_t>(raw_i);
      const auto j = static_cast<std::size_t>(raw_j);
      for (std::size_t k = 0; k < nz; ++k) {
        const auto index = (i * ny + j) * nz + k;
        const auto value = labels[index];
        if (i + 1 < nx && value != labels[((i + 1) * ny + j) * nz + k]) {
          ++count_x;
        }
        if (j + 1 < ny && value != labels[(i * ny + (j + 1)) * nz + k]) {
          ++count_y;
        }
        if (k + 1 < nz && value != labels[index + 1]) {
          ++count_z;
        }
      }
    }
  }
  return {count_x, count_y, count_z};
}

std::vector<std::int64_t> hex_connectivity_range(
    const std::size_t nx,
    const std::size_t ny,
    const std::size_t nz,
    const std::uint64_t start,
    const std::uint64_t stop) {
  const std::uint64_t total = static_cast<std::uint64_t>(nx) * ny * nz;
  if (start > stop || stop > total) {
    throw std::invalid_argument("invalid voxel range");
  }
  const auto rows = static_cast<std::size_t>(stop - start);
  std::vector<std::int64_t> output(rows * 8);
  const std::uint64_t yz_cells = static_cast<std::uint64_t>(ny) * nz;
  const std::uint64_t yz_nodes = static_cast<std::uint64_t>(ny + 1) * (nz + 1);
#pragma omp parallel for schedule(static) if(rows > 100000)
  for (std::int64_t raw_row = 0;
       raw_row < static_cast<std::int64_t>(rows); ++raw_row) {
    const auto row = static_cast<std::uint64_t>(raw_row);
    const auto cell = start + row;
    const auto i = cell / yz_cells;
    const auto remainder = cell % yz_cells;
    const auto j = remainder / nz;
    const auto k = remainder % nz;
    const auto base = i * yz_nodes + j * (nz + 1) + k;
    const auto n100 = base + yz_nodes;
    const auto n010 = base + (nz + 1);
    const auto n110 = n100 + (nz + 1);
    // COMSOL first-order Hex8 tensor-product order:
    // 000, 100, 010, 110, 001, 101, 011, 111.  Do not use the
    // VTK/NASTRAN cyclic order (000,100,110,010,...): COMSOL interprets
    // that permutation as an invalid element and can report adjacent cells
    // on the same side of their shared face.
    const std::array<std::uint64_t, 8> nodes{
        base, n100, n010, n110, base + 1, n100 + 1, n010 + 1, n110 + 1};
    for (std::size_t local = 0; local < 8; ++local) {
      output[static_cast<std::size_t>(row) * 8 + local] =
          static_cast<std::int64_t>(nodes[local]);
    }
  }
  return output;
}

}  // namespace zynnova::zynsim
