/**
 * @file pypbn.cpp
 * @brief provides Python binding definitions
*/

#include "fembv_bin_linear.hpp"
#include "femh1_bin_linear_hmc.hpp"
#include "femh1_bin_linear_mh.hpp"
#include "local_linear_model_ipopt_solver.hpp"

#include <Eigen/Core>

#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using namespace pypbn;

PYBIND11_MODULE(pypbn_ext, m) {
   py::enum_<Ipopt_initial_guess>(m, "IpoptInitialGuess", py::arithmetic())
      .value("Uniform", Ipopt_initial_guess::Uniform)
      .value("Random", Ipopt_initial_guess::Random)
      .value("Current", Ipopt_initial_guess::Current)
      .export_values();

   py::class_<FEMBVBinLinear>(m, "FEMBVBinLinear")
      .def(py::init<
            const Eigen::Ref<const Eigen::VectorXd>,
            const Eigen::Ref<const Eigen::MatrixXd>,
            const Eigen::Ref<const Eigen::MatrixXd>,
            const Eigen::Ref<const Eigen::MatrixXd>,
            double, double, double,
            Ipopt_initial_guess,
            int, int,
            int, int>(),
            py::arg("outcomes"),
            py::arg("predictors"),
            py::arg("parameters"),
            py::arg("affiliations"),
            py::arg("epsilon") = 0.0,
            py::arg("max_tv_norm") = -1.0,
            py::arg("parameters_tolerance") = 1.0e-4,
            py::arg("parameters_initialization") = Ipopt_initial_guess::Uniform,
            py::arg("max_parameters_iterations") = 1000,
            py::arg("max_affiliations_iterations") = 1000,
            py::arg("verbosity") = 0,
            py::arg("random_seed") = 0)
      .def("update_parameters", &FEMBVBinLinear::update_parameters)
      .def("update_affiliations", &FEMBVBinLinear::update_affiliations)
      .def("get_parameters", &FEMBVBinLinear::get_parameters,
         py::return_value_policy::copy)
      .def("get_affiliations", &FEMBVBinLinear::get_affiliations,
         py::return_value_policy::copy)
      .def("get_cost", &FEMBVBinLinear::get_cost)
      .def("get_log_likelihood_bound",
           &FEMBVBinLinear::get_log_likelihood_bound);

   py::class_<FEMH1BinLinearMH>(m, "FEMH1BinLinearMH")
      .def(py::init<
           const Eigen::Ref<const Eigen::VectorXd>,
           const Eigen::Ref<const Eigen::MatrixXd>,
           const Eigen::Ref<const Eigen::MatrixXd>,
           const Eigen::Ref<const Eigen::MatrixXd>,
           double, double, double, double, bool, double,
           Ipopt_initial_guess, int, int, int>(),
           py::arg("outcomes"),
           py::arg("predictors"),
           py::arg("parameters"),
           py::arg("affiliations"),
           py::arg("epsilon_theta") = 0.0,
           py::arg("epsilon_gamma") = 1e-6,
           py::arg("sigma_theta") = 1e-3,
           py::arg("sigma_gamma") = 1e-3,
           py::arg("include_parameters") = true,
           py::arg("parameters_tolerance") = 1e-4,
           py::arg("parameters_initialization") = Ipopt_initial_guess::Uniform,
           py::arg("max_parameters_iterations") = 1000,
           py::arg("verbosity") = 0,
           py::arg("random_seed") = 0)
      .def("metropolis_step", &FEMH1BinLinearMH::metropolis_step)
      .def("get_parameters", &FEMH1BinLinearMH::get_parameters,
           py::return_value_policy::copy)
      .def("get_affiliations", &FEMH1BinLinearMH::get_affiliations,
           py::return_value_policy::copy)
      .def("get_log_likelihood", &FEMH1BinLinearMH::get_log_likelihood)
      .def("get_log_posterior", &FEMH1BinLinearMH::get_log_posterior)
      .def("reset", &FEMH1BinLinearMH::reset)
      .def("get_affiliations_acceptance_rate",
           &FEMH1BinLinearMH::get_affiliations_acceptance_rate)
      .def("get_model_acceptance_rates",
           &FEMH1BinLinearMH::get_model_acceptance_rates,
           py::return_value_policy::copy);

   py::class_<FEMH1BinLinearHMC>(m, "FEMH1BinLinearHMC")
      .def(py::init<
           const Eigen::Ref<const Eigen::VectorXd>,
           const Eigen::Ref<const Eigen::MatrixXd>,
           const Eigen::Ref<const Eigen::MatrixXd>,
           const Eigen::Ref<const Eigen::MatrixXd>,
           double, double, int, double, int, int>(),
           py::arg("outcomes"),
           py::arg("predictors"),
           py::arg("parameters"),
           py::arg("affiliations"),
           py::arg("epsilon_theta") = 0.0,
           py::arg("epsilon_gamma") = 1e-6,
           py::arg("n_leapfrog_steps") = 10,
           py::arg("leapfrog_step_size") = 0.001,
           py::arg("verbosity") = 0,
           py::arg("random_seed") = 0)
      .def("hmc_step", &FEMH1BinLinearHMC::hmc_step)
      .def("get_parameters", &FEMH1BinLinearHMC::get_parameters,
           py::return_value_policy::copy)
      .def("get_affiliations", &FEMH1BinLinearHMC::get_affiliations,
           py::return_value_policy::copy)
      .def("get_log_likelihood", &FEMH1BinLinearHMC::get_log_likelihood)
      .def("get_log_posterior", &FEMH1BinLinearHMC::get_log_posterior)
      .def("reset", &FEMH1BinLinearHMC::reset)
      .def("get_acceptance_rate", &FEMH1BinLinearHMC::get_acceptance_rate);
}
