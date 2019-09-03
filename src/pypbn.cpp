/**
 * @file pypbn.cpp
 * @brief provides Python binding definitions
*/

#include "fembv_bin_linear.hpp"
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
      .value("Current", Ipopt_initial_guess::Current);

   py::class_<FEMBVBinLinear>(m, "FEMBVBinLinear")
      .def(py::init<
            const Eigen::Ref<const Eigen::MatrixXd>&,
            const Eigen::Ref<const Eigen::MatrixXd>&,
            const Eigen::Ref<const Eigen::MatrixXd>&,
            const Eigen::Ref<const Eigen::MatrixXd>&,
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
      .def("get_log_likelihood_bound", &FEMBVBinLinear::get_log_likelihood_bound);
}