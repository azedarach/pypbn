#ifndef PYPBN_FEMH1_BIN_LINEAR_HMC_HPP_INCLUDED
#define PYPBN_FEMH1_BIN_LINEAR_HMC_HPP_INCLUDED

#include "femh1_bin_linear_distributions.hpp"
#include "local_linear_model.hpp"

#include <Eigen/Core>

#include <random>
#include <vector>

namespace pypbn {

class FEMH1BinLinearHMC {
public:
   FEMH1BinLinearHMC(
      const Eigen::Ref<const Eigen::VectorXd>,
      const Eigen::Ref<const Eigen::MatrixXd>,
      const Eigen::Ref<const Eigen::MatrixXd>,
      const Eigen::Ref<const Eigen::MatrixXd>,
      double, double,
      int, double,
      int, int);

   Eigen::MatrixXd get_parameters() const;
   Eigen::MatrixXd get_affiliations() const;
   double get_log_likelihood() const;

   bool hmc_step();
   void reset();
   void set_number_of_leapfrog_steps(int n) { n_leapfrog_steps = n; }
   void set_leapfrog_step_size(double eps) { leapfrog_step_size = eps; }

   double get_acceptance_rate() const {
      return acceptance_rate;
   }

private:
   using Parameters_prior = FEMH1BinLinear_parameters_exponential_prior;
   using Affiliations_prior = FEMH1BinLinear_softmax_affiliations_normal_prior;

   std::mt19937 generator;

   Eigen::VectorXd outcomes;
   Eigen::MatrixXd predictors;

   Parameters_prior parameters_prior{};
   Affiliations_prior softmax_affiliations_prior{};

   std::vector<Local_linear_model> models;
   Eigen::MatrixXd log_affiliations;

   std::vector<Local_linear_model> temp_models;
   Eigen::MatrixXd temp_log_affiliations;

   double current_energy{0};
   Eigen::VectorXd positions;
   Eigen::VectorXd momenta;
   Eigen::VectorXd current_energy_gradient;
   Eigen::VectorXd new_energy_gradient;

   int verbosity{0};
   int n_leapfrog_steps{10};
   double leapfrog_step_size{0.001};
   int chain_length{0};
   double acceptance_rate{0};
};

} // namespace pypbn

#endif
