#include "femh1_bin_linear_distributions.hpp"
#include "densities.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pypbn {

namespace {

double inverse_simplex_volume(std::size_t np1)
{
   double fact = 1.;
   for (std::size_t i = 2; i <= np1; ++i) {
      fact *= i;
   }
   return fact;
}

} // anonymous namespace

double FEMH1BinLinear_parameters_uniform_prior::operator()(
   const Local_linear_model& model) const
{
   return std::exp(log_value(model));
}

double FEMH1BinLinear_parameters_uniform_prior::log_value(
   const Local_linear_model& model) const
{
   const std::vector<double>& parameters = model.get_parameters();
   const std::size_t n_parameters = parameters.size();

   double sum = 0;
   for (std::size_t i = 0; i < n_parameters; ++i) {
      if (parameters[i] < 0) {
         return -std::numeric_limits<double>::max();
      }

      sum += parameters[i];
   }

   if (sum <= 0 || sum >= 1) {
      return -std::numeric_limits<double>::max();
   }

   return std::log(inverse_simplex_volume(n_parameters));
}

double FEMH1BinLinear_parameters_exponential_prior::operator()(
   const Local_linear_model& model) const
{
   return std::exp(log_value(model));
}

double FEMH1BinLinear_parameters_exponential_prior::log_value(
   const Local_linear_model& model) const
{
   const std::vector<double>& parameters = model.get_parameters();
   const std::size_t n_parameters = parameters.size();

   double sum = 0;
   for (std::size_t i = 0; i < n_parameters; ++i) {
      if (parameters[i] < 0) {
         return -std::numeric_limits<double>::max();
      }

      sum += parameters[i];
   }

   if (sum <= 0 || sum >= 1) {
      return -std::numeric_limits<double>::max();
   }

   if (alpha == 0) {
      return inverse_simplex_volume(n_parameters);
   } else {
      return -alpha * sum;
   }
}

double FEMH1BinLinear_softmax_affiliations_normal_prior::operator()(
   const Eigen::MatrixXd& softmax_affiliations) const
{
   return std::exp(log_value(softmax_affiliations));
}

double FEMH1BinLinear_softmax_affiliations_normal_prior::log_value(
   const Eigen::MatrixXd& softmax_affiliations) const
{
   const int n_samples = softmax_affiliations.cols();

   const Eigen::VectorXd gamma(softmax(softmax_affiliations.col(0)));
   double value = log_dirichlet_density(gamma, alpha, tolerance);

   for (int t = 1; t < n_samples; ++t) {
      value += log_normal_density(softmax_affiliations.col(t),
                                  softmax_affiliations.col(t - 1),
                                  inverse_covariance);
   }

   return value;
}

double femh1_bin_linear_log_likelihood(
   const Eigen::VectorXd& outcomes, const Eigen::MatrixXd& predictors,
   const std::vector<Local_linear_model>& models,
   const Eigen::MatrixXd& softmax_affiliations)
{
   double log_like = 0;

   const int n_features = predictors.rows();
   const int n_samples = softmax_affiliations.cols();
   const int n_components = softmax_affiliations.rows();

   Eigen::MatrixXd local_theta(
      Eigen::MatrixXd::Zero(n_features, n_components));
   for (int i = 0; i < n_components; ++i) {
      const std::vector<int>& predictor_indices =
         models[i].get_predictor_indices();
      const std::vector<double>& parameters = models[i].get_parameters();
      const int n_parameters = parameters.size();
      for (int j = 0; j < n_parameters; ++j) {
         local_theta(predictor_indices[j], i) = parameters[j];
      }
   }

   for (int t = 0; t < n_samples; ++t) {
      Eigen::VectorXd gamma(softmax(softmax_affiliations.col(t)));

      Eigen::VectorXd theta(Eigen::VectorXd::Zero(n_features));
      for (int i = 0; i < n_components; ++i) {
         theta += gamma(i) * local_theta.col(i);
      }

      const double p = theta.dot(predictors.col(t));

      log_like += outcomes(t) * std::log(p)
         + (1 - outcomes(t)) * std::log(1 - p);
   }

   return log_like;
}

double femh1_bin_linear_penalized_log_likelihood(
   const Eigen::VectorXd& outcomes, const Eigen::MatrixXd& predictors,
   const std::vector<Local_linear_model>& models,
   const Eigen::MatrixXd& softmax_affiliations)
{
   double log_like = femh1_bin_linear_log_likelihood(
      outcomes, predictors, models, softmax_affiliations);

   const std::size_t n_components = models.size();
   for (std::size_t i = 0; i < n_components; ++i) {
      log_like -= models[i].regularization();
   }

   return log_like;
}

} // namespace pypbn
