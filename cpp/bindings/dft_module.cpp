#include <zynnova/dft/aimd.hpp>
#include <zynnova/dft/quantum.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using zynnova::dft::AIMDIntegrator;

namespace {

template <typename Value>
std::vector<Value> read_vector(
    const py::array_t<Value, py::array::c_style | py::array::forcecast>& array,
    const char* name
) {
    if (array.ndim() != 1) {
        throw std::invalid_argument(std::string(name) + " must have shape [N]");
    }
    const auto view = array.template unchecked<1>();
    std::vector<Value> result(static_cast<std::size_t>(array.shape(0)));
    for (py::ssize_t i = 0; i < array.shape(0); ++i) {
        result[static_cast<std::size_t>(i)] = view(i);
    }
    return result;
}

std::vector<bool> read_bool_vector(
    const py::array_t<bool, py::array::c_style | py::array::forcecast>& array,
    const char* name
) {
    if (array.ndim() != 1) {
        throw std::invalid_argument(std::string(name) + " must have shape [N]");
    }
    const auto view = array.unchecked<1>();
    std::vector<bool> result(static_cast<std::size_t>(array.shape(0)));
    for (py::ssize_t i = 0; i < array.shape(0); ++i) {
        result[static_cast<std::size_t>(i)] = view(i);
    }
    return result;
}

std::vector<double> read_matrix3(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& array,
    const char* name
) {
    if (array.ndim() != 2 || array.shape(1) != 3) {
        throw std::invalid_argument(std::string(name) + " must have shape [N, 3]");
    }
    const auto view = array.unchecked<2>();
    std::vector<double> result(
        static_cast<std::size_t>(array.shape(0) * array.shape(1))
    );
    for (py::ssize_t atom = 0; atom < array.shape(0); ++atom) {
        for (py::ssize_t component = 0; component < 3; ++component) {
            result[static_cast<std::size_t>(3 * atom + component)] =
                view(atom, component);
        }
    }
    return result;
}

py::array_t<double> matrix3_array(const std::vector<double>& values) {
    const py::ssize_t atom_count =
        static_cast<py::ssize_t>(values.size() / 3);
    py::array_t<double> result(
        py::array::ShapeContainer{atom_count, static_cast<py::ssize_t>(3)}
    );
    auto output = result.mutable_unchecked<2>();
    for (py::ssize_t atom = 0; atom < atom_count; ++atom) {
        for (py::ssize_t component = 0; component < 3; ++component) {
            output(atom, component) =
                values[static_cast<std::size_t>(3 * atom + component)];
        }
    }
    return result;
}

}  // namespace

PYBIND11_MODULE(_dft_native, module) {
    module.doc() =
        "ZynNova native quantum-mechanics and ab-initio MD numerical core";

    module.attr("FORCE_TO_ACCELERATION") =
        zynnova::dft::force_to_acceleration;
    module.attr("KINETIC_ENERGY_FACTOR") =
        zynnova::dft::kinetic_energy_factor;
    module.attr("BOLTZMANN_EV_PER_K") =
        zynnova::dft::boltzmann_eV_per_K;

    module.def(
        "solve_schrodinger_1d",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& grid,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& potential,
           double mass,
           std::size_t num_states,
           double tolerance,
           std::size_t max_iterations) {
            const auto grid_values = read_vector(grid, "grid");
            const auto potential_values = read_vector(potential, "potential");
            zynnova::dft::StationaryResult result;
            {
                py::gil_scoped_release release;
                result = zynnova::dft::solve_schrodinger_1d(
                    grid_values,
                    potential_values,
                    mass,
                    num_states,
                    tolerance,
                    max_iterations
                );
            }

            py::array_t<double> energies(
                static_cast<py::ssize_t>(result.num_states)
            );
            py::array_t<double> wavefunctions(
                py::array::ShapeContainer{
                    static_cast<py::ssize_t>(result.num_states),
                    static_cast<py::ssize_t>(result.num_points)
                }
            );
            py::array_t<double> residuals(
                static_cast<py::ssize_t>(result.num_states)
            );
            auto energies_out = energies.mutable_unchecked<1>();
            auto wavefunctions_out = wavefunctions.mutable_unchecked<2>();
            auto residuals_out = residuals.mutable_unchecked<1>();
            for (std::size_t state = 0; state < result.num_states; ++state) {
                energies_out(static_cast<py::ssize_t>(state)) =
                    result.energies[state];
                residuals_out(static_cast<py::ssize_t>(state)) =
                    result.residual_norms[state];
                for (std::size_t point = 0; point < result.num_points; ++point) {
                    wavefunctions_out(
                        static_cast<py::ssize_t>(state),
                        static_cast<py::ssize_t>(point)
                    ) = result.wavefunctions[state * result.num_points + point];
                }
            }
            py::dict output;
            output["energies"] = std::move(energies);
            output["wavefunctions"] = std::move(wavefunctions);
            output["residual_norms"] = std::move(residuals);
            return output;
        },
        py::arg("grid"),
        py::arg("potential"),
        py::arg("mass") = 1.0,
        py::arg("num_states") = 6,
        py::arg("tolerance") = 1.0e-12,
        py::arg("max_iterations") = 80
    );

    module.def(
        "propagate_schrodinger_1d",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& grid,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& potential,
           const py::array_t<
               std::complex<double>,
               py::array::c_style | py::array::forcecast>& initial_wavefunction,
           double mass,
           double timestep,
           std::size_t steps,
           std::size_t save_every) {
            const auto grid_values = read_vector(grid, "grid");
            const auto potential_values = read_vector(potential, "potential");
            const auto wavefunction_values =
                read_vector(initial_wavefunction, "initial_wavefunction");
            zynnova::dft::PropagationResult result;
            {
                py::gil_scoped_release release;
                result = zynnova::dft::propagate_schrodinger_1d(
                    grid_values,
                    potential_values,
                    wavefunction_values,
                    mass,
                    timestep,
                    steps,
                    save_every
                );
            }

            py::array_t<double> times(
                static_cast<py::ssize_t>(result.num_frames)
            );
            py::array_t<double> norms(
                static_cast<py::ssize_t>(result.num_frames)
            );
            py::array_t<std::complex<double>> wavefunctions(
                py::array::ShapeContainer{
                    static_cast<py::ssize_t>(result.num_frames),
                    static_cast<py::ssize_t>(result.num_points)
                }
            );
            auto times_out = times.mutable_unchecked<1>();
            auto norms_out = norms.mutable_unchecked<1>();
            auto wavefunctions_out = wavefunctions.mutable_unchecked<2>();
            for (std::size_t frame = 0; frame < result.num_frames; ++frame) {
                times_out(static_cast<py::ssize_t>(frame)) =
                    result.times[frame];
                norms_out(static_cast<py::ssize_t>(frame)) =
                    result.norms[frame];
                for (std::size_t point = 0; point < result.num_points; ++point) {
                    wavefunctions_out(
                        static_cast<py::ssize_t>(frame),
                        static_cast<py::ssize_t>(point)
                    ) = result.wavefunctions[frame * result.num_points + point];
                }
            }
            py::dict output;
            output["times"] = std::move(times);
            output["wavefunctions"] = std::move(wavefunctions);
            output["norms"] = std::move(norms);
            return output;
        },
        py::arg("grid"),
        py::arg("potential"),
        py::arg("initial_wavefunction"),
        py::arg("mass") = 1.0,
        py::arg("timestep") = 0.01,
        py::arg("steps") = 1,
        py::arg("save_every") = 1
    );

    module.def(
        "maxwell_boltzmann_velocities",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& masses,
           const py::array_t<bool, py::array::c_style | py::array::forcecast>& mobile,
           double temperature_K,
           std::uint64_t seed,
           bool remove_center_of_mass,
           bool exact_temperature) {
            auto result = zynnova::dft::maxwell_boltzmann_velocities(
                read_vector(masses, "masses"),
                read_bool_vector(mobile, "mobile"),
                temperature_K,
                seed,
                remove_center_of_mass,
                exact_temperature
            );
            return matrix3_array(result);
        },
        py::arg("masses"),
        py::arg("mobile"),
        py::arg("temperature_K"),
        py::arg("seed") = 0,
        py::arg("remove_center_of_mass") = true,
        py::arg("exact_temperature") = true
    );

    module.def(
        "kinetic_energy_eV",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& masses,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& velocities,
           const py::array_t<bool, py::array::c_style | py::array::forcecast>& mobile) {
            return zynnova::dft::kinetic_energy_eV(
                read_vector(masses, "masses"),
                read_matrix3(velocities, "velocities"),
                read_bool_vector(mobile, "mobile")
            );
        },
        py::arg("masses"),
        py::arg("velocities"),
        py::arg("mobile")
    );

    module.def(
        "instantaneous_temperature_K",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& masses,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& velocities,
           const py::array_t<bool, py::array::c_style | py::array::forcecast>& mobile,
           bool remove_center_of_mass_dof) {
            return zynnova::dft::instantaneous_temperature_K(
                read_vector(masses, "masses"),
                read_matrix3(velocities, "velocities"),
                read_bool_vector(mobile, "mobile"),
                remove_center_of_mass_dof
            );
        },
        py::arg("masses"),
        py::arg("velocities"),
        py::arg("mobile"),
        py::arg("remove_center_of_mass_dof") = true
    );

    py::class_<AIMDIntegrator>(module, "AIMDIntegrator")
        .def(
            py::init(
                [](const py::array_t<
                       double,
                       py::array::c_style | py::array::forcecast>& masses,
                   const py::array_t<
                       bool,
                       py::array::c_style | py::array::forcecast>& mobile,
                   double timestep_fs,
                   const std::string& ensemble,
                   double temperature_K,
                   double friction_per_fs,
                   std::uint64_t seed) {
                    return std::make_unique<AIMDIntegrator>(
                        read_vector(masses, "masses"),
                        read_bool_vector(mobile, "mobile"),
                        timestep_fs,
                        ensemble,
                        temperature_K,
                        friction_per_fs,
                        seed
                    );
                }
            ),
            py::arg("masses"),
            py::arg("mobile"),
            py::arg("timestep_fs"),
            py::arg("ensemble") = "nve",
            py::arg("temperature_K") = 300.0,
            py::arg("friction_per_fs") = 0.01,
            py::arg("seed") = 0
        )
        .def(
            "set_velocities",
            [](AIMDIntegrator& self,
               const py::array_t<
                   double,
                   py::array::c_style | py::array::forcecast>& velocities) {
                self.set_velocities(read_matrix3(velocities, "velocities"));
            },
            py::arg("velocities")
        )
        .def(
            "velocities",
            [](const AIMDIntegrator& self) {
                return matrix3_array(self.velocities());
            }
        )
        .def(
            "begin_step",
            [](AIMDIntegrator& self,
               const py::array_t<
                   double,
                   py::array::c_style | py::array::forcecast>& positions,
               const py::array_t<
                   double,
                   py::array::c_style | py::array::forcecast>& forces) {
                const auto positions_values = read_matrix3(positions, "positions");
                const auto forces_values = read_matrix3(forces, "forces");
                std::vector<double> result;
                {
                    py::gil_scoped_release release;
                    result = self.begin_step(positions_values, forces_values);
                }
                return matrix3_array(result);
            },
            py::arg("positions"),
            py::arg("forces")
        )
        .def(
            "end_step",
            [](AIMDIntegrator& self,
               const py::array_t<
                   double,
                   py::array::c_style | py::array::forcecast>& forces) {
                const auto forces_values = read_matrix3(forces, "forces");
                std::vector<double> result;
                {
                    py::gil_scoped_release release;
                    result = self.end_step(forces_values);
                }
                return matrix3_array(result);
            },
            py::arg("forces")
        )
        .def("kinetic_energy_eV", &AIMDIntegrator::kinetic_energy_eV)
        .def(
            "temperature_K",
            &AIMDIntegrator::temperature_K,
            py::arg("remove_center_of_mass_dof") = true
        )
        .def_property(
            "step_index",
            &AIMDIntegrator::step_index,
            &AIMDIntegrator::set_step_index
        )
        .def_property_readonly(
            "awaiting_forces", &AIMDIntegrator::awaiting_forces
        )
        .def_property_readonly("ensemble", &AIMDIntegrator::ensemble)
        .def_property(
            "rng_state",
            &AIMDIntegrator::rng_state,
            &AIMDIntegrator::set_rng_state
        );
}
