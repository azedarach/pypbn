#ifndef PYPBN_FEMBV_BIN_LINEAR_HPP_INCLUDED
#define PYPBN_FEMBV_BIN_LINEAR_HPP_INCLUDED

#include "clpsimplex_affiliations_solver.hpp"
#include "local_linear_model_ipopt_solver.hpp"

#include <Eigen/Core>

#include <tuple>
#include <vector>

namespace pypbn {

class FEMBVBinLinear {
public:
   FEMBVBinLinear(
      const Eigen::Ref<const Eigen::VectorXd>,
      const Eigen::Ref<const Eigen::MatrixXd>,
      const Eigen::Ref<const Eigen::MatrixXd>,
      const Eigen::Ref<const Eigen::MatrixXd>,
      double, double, double,
      Ipopt_initial_guess,
      int, int,
      int, int);
   ~FEMBVBinLinear() = default;

   Eigen::MatrixXd get_parameters() const;
   const Eigen::MatrixXd& get_affiliations() const { return affiliations; }
   double get_cost();
   double get_log_likelihood_bound() const;

   bool update_parameters();
   bool update_affiliations();

private:
   Local_linear_model_ipopt_solver theta_solver;
   ClpSimplex_affiliations_solver gamma_solver;

   Eigen::VectorXd outcomes;
   Eigen::MatrixXd predictors;
   Eigen::MatrixXd basis_values;
   Eigen::MatrixXd distances;

   std::vector<Local_linear_model> models{};
   Eigen::MatrixXd affiliations;

   double calculate_log_likelihood_bound() const;
};

} // namespace pypbn

#endif
