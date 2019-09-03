#include "fembv_bin_linear.hpp"
#include "clpsimplex_affiliations_solver.hpp"
#include "local_linear_model.hpp"

#include <iostream>
#include <stdexcept>

namespace pypbn {

namespace {

double fembv_bin_cost(
   const Eigen::MatrixXd& Gamma, const std::vector<Local_linear_model>& models,
   const Eigen::MatrixXd& G)
{
   double cost = (Gamma.transpose() * G).trace();
   for (const auto& m: models) {
      cost += m.regularization();
   }
   return cost;
}

void fill_fembv_bin_distance_matrix(
   const Eigen::VectorXd& Y, const Eigen::MatrixXd& X,
   const std::vector<Local_linear_model>& models, Eigen::MatrixXd& G)
{
   const int n_samples = X.cols();
   const int n_components = models.size();

   for (int t = 0; t < n_samples; ++t) {
      for (int i = 0; i < n_components; ++i) {
         G(i, t) = -models[i].log_likelihood_component(Y(t), X.col(t));
      }
   }
}

bool update_fembv_bin_parameters(
   const Eigen::VectorXd& Y, const Eigen::MatrixXd& X,
   const Eigen::MatrixXd& Gamma,
   std::vector<Local_linear_model>& models,
   Local_linear_model_ipopt_solver& solver)
{
   const int n_components = Gamma.rows();

   bool success = true;
   for (int i = 0; i < n_components; ++i) {
      success = success && solver.update_local_model(
         Y, X, Gamma.row(i), models[i]);
   }

   return success;
}

} // anonymous namespace

FEMBVBinLinear::FEMBVBinLinear(
   const Eigen::Ref<const Eigen::MatrixXd>& outcomes_,
   const Eigen::Ref<const Eigen::MatrixXd>& predictors_,
   const Eigen::Ref<const Eigen::MatrixXd>& parameters_,
   const Eigen::Ref<const Eigen::MatrixXd>& affiliations_,
   double epsilon, double max_tv_norm, double parameters_tolerance,
   Ipopt_initial_guess parameters_initialization,
   int max_parameters_iterations, int max_affiliations_iterations,
   int verbosity, int random_seed)
   : theta_solver(random_seed)
   , gamma_solver(Eigen::MatrixXd::Zero(parameters_.rows(), predictors_.cols()),
                  Eigen::MatrixXd::Identity(predictors_.cols(), predictors_.cols()),
                  max_tv_norm)
{
   const int n_samples = predictors_.cols();
   const int n_components = parameters_.rows();
   const int n_features = predictors_.rows();

   if (affiliations_.cols() != n_samples) {
      throw std::runtime_error(
         "number of affiliation samples does not match "
         "number of data samples");
   }

   if (affiliations_.rows() != n_components) {
      throw std::runtime_error(
         "number of affiliation series does not match "
         "number of components");
   }

   if (parameters_.cols() != n_features) {
      throw std::runtime_error(
         "number of parameters does not match "
         "number of features");
   }

   basis_values = Eigen::MatrixXd::Identity(n_samples, n_samples);
   distances = Eigen::MatrixXd::Zero(n_components, n_samples);

   theta_solver.set_tolerance(parameters_tolerance);
   theta_solver.set_max_iterations(max_parameters_iterations);
   theta_solver.set_verbosity(verbosity);
   theta_solver.set_initialization_method(parameters_initialization);
   theta_solver.initialize();

   gamma_solver.set_max_iterations(max_affiliations_iterations);
   gamma_solver.set_verbosity(verbosity);

   outcomes = outcomes_;
   predictors = predictors_;
   affiliations = affiliations_;

   models = std::vector<Local_linear_model>(n_components);
   for (int i = 0; i < n_components; ++i) {
      models[i] = Local_linear_model(n_features);
      std::vector<double> theta(n_features);
      for (int j = 0; j < n_features; ++j) {
         theta[j] = parameters_(i, j);
      }
      models[i].set_parameters(theta);
      models[i].epsilon = epsilon;
   }
}

bool FEMBVBinLinear::update_parameters()
{
   return update_fembv_bin_parameters(
      outcomes, predictors, affiliations, models, theta_solver);
}

bool FEMBVBinLinear::update_affiliations()
{
   fill_fembv_bin_distance_matrix(
      outcomes, predictors, models, distances);

   const auto gamma_status = gamma_solver.update_affiliations(
      distances);

   bool affiliations_success;
   if (static_cast<int>(gamma_status) == 0) {
      affiliations_success = true;
   } else {
      affiliations_success = false;
   }
      
   gamma_solver.get_affiliations(affiliations);

   return affiliations_success;
}

Eigen::MatrixXd FEMBVBinLinear::get_parameters() const
{
   const int n_components = models.size();
   const int n_features = predictors.rows();

   Eigen::MatrixXd parameters(Eigen::MatrixXd::Zero(n_components, n_features));
   for (int i = 0; i < n_components; ++i) {
      const std::vector<double>& theta = models[i].get_parameters();
      for (int j = 0; j < n_features; ++j) {
         parameters(i, j) = theta[j];
      }
   }

   return parameters;
}

double FEMBVBinLinear::calculate_log_likelihood_bound() const
{
   using std::log;

   const int n_components = models.size();
   const int n_samples = predictors.cols();

   double bound = 0;
   for (int t = 0; t < n_samples; ++t) {
      for (int i = 0; i < n_components; ++i) {
         bound += affiliations(i, t) * models[i].log_likelihood_component(
            outcomes(t), predictors.col(t));
      }
   }

   return bound;
}

double FEMBVBinLinear::get_cost()
{
   fill_fembv_bin_distance_matrix(
      outcomes, predictors, models, distances);

   return fembv_bin_cost(affiliations, models, distances);
}

double FEMBVBinLinear::get_log_likelihood_bound() const
{
   return calculate_log_likelihood_bound();
}

} // namespace pypbn
