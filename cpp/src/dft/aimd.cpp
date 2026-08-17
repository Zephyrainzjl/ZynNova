#include <zynnova/dft/aimd.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace zynnova::dft {
namespace {

std::size_t mobile_count(const std::vector<bool>& mobile) {
    return static_cast<std::size_t>(
        std::count(mobile.begin(), mobile.end(), true)
    );
}

void validate_system(
    const std::vector<double>& masses,
    const std::vector<bool>& mobile
) {
    if (masses.empty()) {
        throw std::invalid_argument("masses must contain at least one atom");
    }
    if (mobile.size() != masses.size()) {
        throw std::invalid_argument("mobile must have the same length as masses");
    }
    if (mobile_count(mobile) == 0) {
        throw std::invalid_argument("at least one atom must be mobile");
    }
    for (double mass : masses) {
        if (!std::isfinite(mass) || mass <= 0.0) {
            throw std::invalid_argument("all masses must be finite and positive");
        }
    }
}

void validate_phase_space_vector(
    const std::vector<double>& values,
    std::size_t atom_count,
    const char* name
) {
    if (values.size() != 3 * atom_count) {
        throw std::invalid_argument(
            std::string(name) + " must have shape [N, 3]"
        );
    }
    if (!std::all_of(values.begin(), values.end(), [](double value) {
            return std::isfinite(value);
        })) {
        throw std::invalid_argument(
            std::string(name) + " must contain only finite values"
        );
    }
}

void remove_center_of_mass_velocity(
    const std::vector<double>& masses,
    const std::vector<bool>& mobile,
    std::vector<double>& velocities
) {
    if (mobile_count(mobile) <= 1) {
        return;
    }
    double total_mass = 0.0;
    double center_of_mass_velocity[3] = {0.0, 0.0, 0.0};
    for (std::size_t atom = 0; atom < masses.size(); ++atom) {
        if (!mobile[atom]) {
            continue;
        }
        total_mass += masses[atom];
        for (std::size_t component = 0; component < 3; ++component) {
            center_of_mass_velocity[component] +=
                masses[atom] * velocities[3 * atom + component];
        }
    }
    for (double& component : center_of_mass_velocity) {
        component /= total_mass;
    }
    for (std::size_t atom = 0; atom < masses.size(); ++atom) {
        if (!mobile[atom]) {
            continue;
        }
        for (std::size_t component = 0; component < 3; ++component) {
            velocities[3 * atom + component] -=
                center_of_mass_velocity[component];
        }
    }
}

}  // namespace

AIMDIntegrator::AIMDIntegrator(
    std::vector<double> masses,
    std::vector<bool> mobile,
    double timestep_fs,
    std::string ensemble,
    double temperature_K,
    double friction_per_fs,
    std::uint64_t seed
)
    : masses_(std::move(masses)),
      mobile_(std::move(mobile)),
      velocities_(3 * masses_.size(), 0.0),
      timestep_fs_(timestep_fs),
      ensemble_(std::move(ensemble)),
      temperature_K_(temperature_K),
      friction_per_fs_(friction_per_fs),
      random_engine_(seed) {
    validate_system(masses_, mobile_);
    if (!std::isfinite(timestep_fs_) || timestep_fs_ <= 0.0) {
        throw std::invalid_argument("timestep_fs must be finite and positive");
    }
    if (ensemble_ != "nve" && ensemble_ != "nvt_langevin") {
        throw std::invalid_argument("ensemble must be 'nve' or 'nvt_langevin'");
    }
    if (ensemble_ == "nvt_langevin"
        && (!std::isfinite(temperature_K_) || temperature_K_ <= 0.0)) {
        throw std::invalid_argument(
            "nvt_langevin requires a finite, positive temperature_K"
        );
    }
    if (!std::isfinite(friction_per_fs_) || friction_per_fs_ < 0.0) {
        throw std::invalid_argument("friction_per_fs must be finite and non-negative");
    }
}

void AIMDIntegrator::validate_vector(
    const std::vector<double>& values,
    const char* name
) const {
    validate_phase_space_vector(values, masses_.size(), name);
}

void AIMDIntegrator::set_velocities(const std::vector<double>& velocities) {
    if (awaiting_forces_) {
        throw std::logic_error("cannot replace velocities in the middle of a step");
    }
    validate_vector(velocities, "velocities");
    velocities_ = velocities;
    for (std::size_t atom = 0; atom < masses_.size(); ++atom) {
        if (!mobile_[atom]) {
            for (std::size_t component = 0; component < 3; ++component) {
                velocities_[3 * atom + component] = 0.0;
            }
        }
    }
}

const std::vector<double>& AIMDIntegrator::velocities() const noexcept {
    return velocities_;
}

void AIMDIntegrator::apply_half_kick(const std::vector<double>& forces) {
    const double factor = 0.5 * timestep_fs_ * force_to_acceleration;
    for (std::size_t atom = 0; atom < masses_.size(); ++atom) {
        if (!mobile_[atom]) {
            continue;
        }
        for (std::size_t component = 0; component < 3; ++component) {
            const std::size_t index = 3 * atom + component;
            velocities_[index] += factor * forces[index] / masses_[atom];
        }
    }
}

std::vector<double> AIMDIntegrator::begin_step(
    const std::vector<double>& positions,
    const std::vector<double>& forces
) {
    if (awaiting_forces_) {
        throw std::logic_error("end_step must be called before beginning another step");
    }
    validate_vector(positions, "positions");
    validate_vector(forces, "forces");
    apply_half_kick(forces);

    std::vector<double> updated_positions = positions;
    if (ensemble_ == "nve") {
        for (std::size_t atom = 0; atom < masses_.size(); ++atom) {
            if (!mobile_[atom]) {
                continue;
            }
            for (std::size_t component = 0; component < 3; ++component) {
                const std::size_t index = 3 * atom + component;
                updated_positions[index] += timestep_fs_ * velocities_[index];
            }
        }
    } else {
        const double half_timestep = 0.5 * timestep_fs_;
        for (std::size_t atom = 0; atom < masses_.size(); ++atom) {
            if (!mobile_[atom]) {
                continue;
            }
            for (std::size_t component = 0; component < 3; ++component) {
                const std::size_t index = 3 * atom + component;
                updated_positions[index] += half_timestep * velocities_[index];
            }
        }

        const double damping = std::exp(-friction_per_fs_ * timestep_fs_);
        const double noise_fraction =
            std::max(0.0, 1.0 - damping * damping);
        std::normal_distribution<double> normal(0.0, 1.0);
        for (std::size_t atom = 0; atom < masses_.size(); ++atom) {
            if (!mobile_[atom]) {
                continue;
            }
            const double sigma = std::sqrt(
                noise_fraction * boltzmann_eV_per_K * temperature_K_
                / (masses_[atom] * kinetic_energy_factor)
            );
            for (std::size_t component = 0; component < 3; ++component) {
                const std::size_t index = 3 * atom + component;
                velocities_[index] =
                    damping * velocities_[index] + sigma * normal(random_engine_);
                updated_positions[index] += half_timestep * velocities_[index];
            }
        }
    }
    awaiting_forces_ = true;
    return updated_positions;
}

std::vector<double> AIMDIntegrator::end_step(const std::vector<double>& forces) {
    if (!awaiting_forces_) {
        throw std::logic_error("begin_step must be called before end_step");
    }
    validate_vector(forces, "forces");
    apply_half_kick(forces);
    awaiting_forces_ = false;
    ++step_index_;
    return velocities_;
}

double AIMDIntegrator::kinetic_energy_eV() const {
    return zynnova::dft::kinetic_energy_eV(masses_, velocities_, mobile_);
}

double AIMDIntegrator::temperature_K(bool remove_center_of_mass_dof) const {
    return instantaneous_temperature_K(
        masses_, velocities_, mobile_, remove_center_of_mass_dof
    );
}

std::size_t AIMDIntegrator::step_index() const noexcept {
    return step_index_;
}

void AIMDIntegrator::set_step_index(std::size_t value) noexcept {
    step_index_ = value;
}

bool AIMDIntegrator::awaiting_forces() const noexcept {
    return awaiting_forces_;
}

const std::string& AIMDIntegrator::ensemble() const noexcept {
    return ensemble_;
}

std::string AIMDIntegrator::rng_state() const {
    std::ostringstream stream;
    stream << random_engine_;
    return stream.str();
}

void AIMDIntegrator::set_rng_state(const std::string& state) {
    if (awaiting_forces_) {
        throw std::logic_error("cannot restore RNG state in the middle of a step");
    }
    std::istringstream stream(state);
    stream >> random_engine_;
    if (!stream) {
        throw std::invalid_argument("invalid random-number-generator state");
    }
}

std::vector<double> maxwell_boltzmann_velocities(
    const std::vector<double>& masses,
    const std::vector<bool>& mobile,
    double temperature_K,
    std::uint64_t seed,
    bool remove_center_of_mass,
    bool exact_temperature
) {
    validate_system(masses, mobile);
    if (!std::isfinite(temperature_K) || temperature_K <= 0.0) {
        throw std::invalid_argument("temperature_K must be finite and positive");
    }

    std::mt19937_64 random_engine(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<double> velocities(3 * masses.size(), 0.0);
    for (std::size_t atom = 0; atom < masses.size(); ++atom) {
        if (!mobile[atom]) {
            continue;
        }
        const double sigma = std::sqrt(
            boltzmann_eV_per_K * temperature_K
            / (masses[atom] * kinetic_energy_factor)
        );
        for (std::size_t component = 0; component < 3; ++component) {
            velocities[3 * atom + component] = sigma * normal(random_engine);
        }
    }
    const bool remove_com = remove_center_of_mass && mobile_count(mobile) > 1;
    if (remove_com) {
        remove_center_of_mass_velocity(masses, mobile, velocities);
    }
    if (exact_temperature) {
        const double current_temperature = instantaneous_temperature_K(
            masses, velocities, mobile, remove_com
        );
        if (!(current_temperature > 0.0) || !std::isfinite(current_temperature)) {
            throw std::runtime_error(
                "cannot rescale a zero-temperature velocity distribution"
            );
        }
        const double scale = std::sqrt(temperature_K / current_temperature);
        for (std::size_t atom = 0; atom < masses.size(); ++atom) {
            if (!mobile[atom]) {
                continue;
            }
            for (std::size_t component = 0; component < 3; ++component) {
                velocities[3 * atom + component] *= scale;
            }
        }
    }
    return velocities;
}

double kinetic_energy_eV(
    const std::vector<double>& masses,
    const std::vector<double>& velocities,
    const std::vector<bool>& mobile
) {
    validate_system(masses, mobile);
    validate_phase_space_vector(velocities, masses.size(), "velocities");
    double energy = 0.0;
    for (std::size_t atom = 0; atom < masses.size(); ++atom) {
        if (!mobile[atom]) {
            continue;
        }
        double speed_squared = 0.0;
        for (std::size_t component = 0; component < 3; ++component) {
            const double velocity = velocities[3 * atom + component];
            speed_squared += velocity * velocity;
        }
        energy += 0.5 * kinetic_energy_factor * masses[atom] * speed_squared;
    }
    return energy;
}

double instantaneous_temperature_K(
    const std::vector<double>& masses,
    const std::vector<double>& velocities,
    const std::vector<bool>& mobile,
    bool remove_center_of_mass_dof
) {
    const std::size_t count = mobile_count(mobile);
    std::size_t degrees_of_freedom = 3 * count;
    if (remove_center_of_mass_dof && count > 1) {
        degrees_of_freedom -= 3;
    }
    if (degrees_of_freedom == 0) {
        return 0.0;
    }
    const double energy = kinetic_energy_eV(masses, velocities, mobile);
    return 2.0 * energy
        / (static_cast<double>(degrees_of_freedom) * boltzmann_eV_per_K);
}

}  // namespace zynnova::dft
