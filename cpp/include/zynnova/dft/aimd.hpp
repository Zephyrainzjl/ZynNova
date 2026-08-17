#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace zynnova::dft {

inline constexpr double force_to_acceleration = 0.009648533215665327;
inline constexpr double kinetic_energy_factor = 103.64269652680505;
inline constexpr double boltzmann_eV_per_K = 8.617333262145e-5;

class AIMDIntegrator {
public:
    AIMDIntegrator(
        std::vector<double> masses,
        std::vector<bool> mobile,
        double timestep_fs,
        std::string ensemble,
        double temperature_K,
        double friction_per_fs,
        std::uint64_t seed
    );

    void set_velocities(const std::vector<double>& velocities);
    [[nodiscard]] const std::vector<double>& velocities() const noexcept;
    [[nodiscard]] std::vector<double> begin_step(
        const std::vector<double>& positions,
        const std::vector<double>& forces
    );
    [[nodiscard]] std::vector<double> end_step(const std::vector<double>& forces);

    [[nodiscard]] double kinetic_energy_eV() const;
    [[nodiscard]] double temperature_K(bool remove_center_of_mass_dof = true) const;
    [[nodiscard]] std::size_t step_index() const noexcept;
    void set_step_index(std::size_t value) noexcept;
    [[nodiscard]] bool awaiting_forces() const noexcept;
    [[nodiscard]] const std::string& ensemble() const noexcept;
    [[nodiscard]] std::string rng_state() const;
    void set_rng_state(const std::string& state);

private:
    void validate_vector(const std::vector<double>& values, const char* name) const;
    void apply_half_kick(const std::vector<double>& forces);

    std::vector<double> masses_;
    std::vector<bool> mobile_;
    std::vector<double> velocities_;
    double timestep_fs_{};
    std::string ensemble_;
    double temperature_K_{};
    double friction_per_fs_{};
    std::mt19937_64 random_engine_;
    std::size_t step_index_{};
    bool awaiting_forces_{false};
};

std::vector<double> maxwell_boltzmann_velocities(
    const std::vector<double>& masses,
    const std::vector<bool>& mobile,
    double temperature_K,
    std::uint64_t seed,
    bool remove_center_of_mass = true,
    bool exact_temperature = true
);

double kinetic_energy_eV(
    const std::vector<double>& masses,
    const std::vector<double>& velocities,
    const std::vector<bool>& mobile
);

double instantaneous_temperature_K(
    const std::vector<double>& masses,
    const std::vector<double>& velocities,
    const std::vector<bool>& mobile,
    bool remove_center_of_mass_dof = true
);

}  // namespace zynnova::dft
