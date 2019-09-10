#ifndef PYPBN_FEMH1_BIN_LINEAR_MH_HPP_INCLUDED
#define PYPBN_FEMH1_BIN_LINEAR_MH_HPP_INCLUDED

#include "femh1_bin_linear_distributions.hpp"
#include "local_linear_model_ipopt_solver.hpp"

#include <Eigen/Core>

#include <random>
#include <vector>

namespace pypbn {

class FEMH1BinLinearMH {
public:
   FEMH1BinLinearMH(
      const Eigen::Ref<const Eigen::VectorXd>,
      const Eigen::Ref<const Eigen::MatrixXd>,
      const Eigen::Ref<const Eigen::MatrixXd>,
      const Eigen::Ref<const Eigen::MatrixXd>,
      double, double, double, double, bool, double,
      Ipopt_initial_guess, int, int, int);
   ~FEMH1BinLinearMH() = default;

   Eigen::MatrixXd get_parameters() const;
   Eigen::MatrixXd get_affiliations() const;
   double get_log_likelihood() const;
   double get_log_posterior() const;

   bool metropolis_step();
   void reset();

   double get_affiliations_acceptance_rate() const {
      return affiliations_acceptance_rate;
   }

   double get_model_acceptance_rate(int) const;
   const std::vector<double>& get_model_acceptance_rates() const {
      return model_acceptance_rates;
   }

private:
   using Parameters_prior = FEMH1BinLinear_parameters_exponential_prior;
   using Affiliations_prior = FEMH1BinLinear_softmax_affiliations_normal_prior;

   Local_linear_model_ipopt_solver theta_solver;
   std::mt19937 generator;

   Eigen::VectorXd outcomes;
   Eigen::MatrixXd predictors;
   Eigen::MatrixXd distances;

   Parameters_prior parameters_prior{};
   Affiliations_prior softmax_affiliations_prior{};

   std::vector<Local_linear_model> models;
   Eigen::MatrixXd log_affiliations;

   bool include_parameters{true};
   double sigma_theta{1e-3};
   double sigma_gamma{1e-3};
   int chain_length{0};
   double affiliations_acceptance_rate{0};
   std::vector<double> model_acceptance_rates{};

   bool update_parameters();
};

} // namespace pypbn

#endif
