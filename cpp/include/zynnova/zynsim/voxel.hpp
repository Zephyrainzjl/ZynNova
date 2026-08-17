#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace zynnova::zynsim {

std::vector<std::int32_t> threshold_labels_u8(
    const std::uint8_t* values,
    std::size_t count,
    const std::vector<std::uint8_t>& lower,
    const std::vector<std::uint8_t>& upper,
    const std::vector<std::int32_t>& labels,
    std::int32_t unclassified);

std::vector<std::int32_t> fuse_orthogonal_labels(
    const std::int32_t* xy,
    const std::int32_t* xz,
    const std::int32_t* yz,
    std::size_t nx,
    std::size_t ny,
    std::size_t nz,
    const std::vector<std::int32_t>& phases,
    const std::vector<double>& log_priors,
    double projection_weight);

std::map<std::int32_t, std::uint64_t> phase_counts(
    const std::int32_t* labels,
    std::size_t count);

std::array<std::uint64_t, 3> interface_counts(
    const std::int32_t* labels,
    std::size_t nx,
    std::size_t ny,
    std::size_t nz);

std::vector<std::int64_t> hex_connectivity_range(
    std::size_t nx,
    std::size_t ny,
    std::size_t nz,
    std::uint64_t start,
    std::uint64_t stop);

}  // namespace zynnova::zynsim
