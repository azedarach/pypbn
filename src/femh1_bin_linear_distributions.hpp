#ifndef PYPBN_FEMH1_BIN_LINEAR_DISTRIBUTIONS_HPP
#define PYPBN_FEMH1_BIN_LINEAR_DISTRIBUTIONS_HPP

#include "densities.hpp"
#include "local_linear_model.hpp"

#include <Eigen/Core>

#include <vector>

namespace pypbn {

struct FEMH1BinLinear_parameters_uniform_prior {
   double operator()(const Local_linear_model&) const;
   double log_value(const Local_linear_model&) const;
   template <class Derived>
   void add_log_gradient(const Local_linear_model&,
                         Eigen::MatrixBase<Derived>&) const;
};

template <class Derived>
void FEMH1BinLinear_parameters_uniform_prior::add_log_gradient(
   const Local_linear_model& /* model */,
   Eigen::MatrixBase<Derived>& /* gradient */) const
{
   // no contribution for uniform prior
}

struct FEMH1BinLinear_parameters_exponential_prior {
   double alpha{0};

   double operator()(const Local_linear_model&) const;
   double log_value(const Local_linear_model&) const;
   template <class Derived>
   void add_log_gradient(const Local_linear_model&,
                         Eigen::MatrixBase<Derived>&) const;
};

template <class Derived>
void FEMH1BinLinear_parameters_exponential_prior::add_log_gradient(
   const Local_linear_model& model,
   Eigen::MatrixBase<Derived>& gradient) const
{
   if (alpha == 0) {
      return;
   }

   const int n_parameters = model.get_number_of_parameters();

   if (gradient.size() != n_parameters) {
      throw std::runtime_error(
         "number of gradient elements does not match number of parameters");
   }

   for (int i = 0; i < n_parameters; ++i) {
      gradient(i) -= alpha;
   }
}

struct FEMH1BinLinear_softmax_affiliations_normal_prior {
   Eigen::VectorXd alpha;
   Eigen::MatrixXd inverse_covariance;
   double tolerance{1e-10};

   double operator()(const Eigen::MatrixXd&) const;
   double log_value(const Eigen::MatrixXd&) const;
   template <class Derived>
   void add_log_gradient(const Eigen::MatrixXd&,
                         Eigen::MatrixBase<Derived>&) const;
};

template <class Derived>
void FEMH1BinLinear_softmax_affiliations_normal_prior::add_log_gradient(
   const Eigen::MatrixXd& softmax_affiliations,
   Eigen::MatrixBase<Derived>& gradient) const
{
   const int n_components = softmax_affiliations.rows();
   const int n_samples = softmax_affiliations.cols();

   const int n_parameters = n_components * n_samples;

   if (gradient.size() != n_parameters) {
      throw std::runtime_error(
         "number of gradient elements does not match number of parameters");
   }

   Eigen::VectorXd v(n_components);
   int gradient_index = 0;
   for (int t = 0; t < n_samples; ++t) {
      if (t < n_samples - 1) {
         v = inverse_covariance * (softmax_affiliations.col(t + 1)
                                   - softmax_affiliations.col(t));
         gradient.segment(gradient_index, n_components) += v;
      }

      if (t > 0) {
         v = inverse_covariance * (softmax_affiliations.col(t)
                                   - softmax_affiliations.col(t - 1));
         gradient.segment(gradient_index, n_components) -= v;
      }

      gradient_index += n_components;
   }
}

double femh1_bin_linear_log_likelihood(
   const Eigen::VectorXd&, const Eigen::MatrixXd&,
   const std::vector<Local_linear_model>&,
   const Eigen::MatrixXd&);

double femh1_bin_linear_penalized_log_likelihood(
   const Eigen::VectorXd&, const Eigen::MatrixXd&,
   const std::vector<Local_linear_model>&,
   const Eigen::MatrixXd&);

template <class Derived>
void add_femh1_bin_linear_log_likelihood_gradient(
   const Eigen::VectorXd& outcomes, const Eigen::MatrixXd& predictors,
   const std::vector<Local_linear_model>& models,
   const Eigen::MatrixXd& softmax_affiliations,
   Eigen::MatrixBase<Derived>& gradient)
{
   const int n_features = predictors.rows();
   const int n_samples = softmax_affiliations.cols();
   const int n_components = softmax_affiliations.rows();

   Eigen::MatrixXd local_theta(
      Eigen::MatrixXd::Zero(n_features, n_components));
   for (int i = 0; i < n_components; ++i) {
      const auto& predictor_indices = models[i].get_predictor_indices();
      const auto& parameters = models[i].get_parameters();
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
      const double y = outcomes(t);
      const double prefactor = (y > 0 ? (1.0 / p) : -1.0 / (1 - p));

      int gradient_index = 0;
      for (int j = 0; j < n_components; ++j) {
         const int n_parameters = models[j].get_number_of_parameters();
         for (int k = 0; k < n_parameters; ++k) {
            gradient(gradient_index) += prefactor * gamma(j) * predictors(k, t);
            ++gradient_index;
         }
      }

      if (n_components > 1) {
         gradient_index += n_components * t;
         for (int j = 0; j < n_components; ++j) {
            const double pj = gamma(j) * local_theta.col(j).dot(
               predictors.col(t));

            gradient(gradient_index) += prefactor * (pj - gamma(j) * p);
         }
      }
   }
}

template <class Derived>
void add_femh1_bin_linear_penalized_log_likelihood_gradient(
   const Eigen::VectorXd& outcomes, const Eigen::MatrixXd& predictors,
   const std::vector<Local_linear_model>& models,
   const Eigen::MatrixXd& softmax_affiliations,
   Eigen::MatrixBase<Derived>& gradient)
{
   add_femh1_bin_linear_log_likelihood_gradient(
      outcomes, predictors, models, softmax_affiliations,
      gradient);

   const std::size_t n_components = models.size();

   int gradient_index = 0;
   for (std::size_t i = 0; i < n_components; ++i) {
      const int n_parameters = models[i].get_number_of_parameters();
      for (int j = 0; j < n_parameters; ++j) {
         gradient(gradient_index) -= models[i].regularization_gradient(j);
         ++gradient_index;
      }
   }
}

} // namespace pypbn

#endif
