#include "catch.hpp"

#include "femh1_bin_linear_distributions.hpp"
#include "local_linear_model.hpp"

#include <Eigen/Core>

#include <vector>

using namespace pypbn;

template <class F, class Vector>
double central_difference_derivative(F f, const Vector& x, int elem, double h)
{
   volatile const double xiph = x(elem) + h;
   volatile const double ximh = x(elem) - h;
   const double dx = xiph - x(elem);

   Vector xph(x);
   xph(elem) = xiph;

   Vector xmh(x);
   xmh(elem) = ximh;

   return (f(xph) - f(xmh)) / (2. * dx);
}

TEST_CASE("Test FEM-H1-BIN linear model uniform priors",
          "[femh1_bin_linear]")
{
   SECTION("Returns zero for parameters outside of simplex")
   {
      const int n_parameters = 3;

      Local_linear_model model(n_parameters);
      std::vector<double> theta(n_parameters, 1);
      model.set_parameters(theta);
      model.epsilon = 1.2;

      FEMH1BinLinear_parameters_uniform_prior prior;

      const double value = prior(model);
      const double expected_value = 0;

      CHECK(value == expected_value);
   }

   SECTION("Returns constant value for parameters inside simplex")
   {
      const int n_parameters = 4;
      const double tolerance = 1.0e-10;

      Local_linear_model model(n_parameters);
      std::vector<double> theta(n_parameters);
      theta[0] = 0.;
      theta[1] = 0.5;
      theta[2] = 0.1;
      theta[3] = 0.1;
      model.set_parameters(theta);
      model.epsilon = 0.5;

      FEMH1BinLinear_parameters_uniform_prior prior;

      const double value = prior(model);
      double expected_value = 1;
      for (int i = 2; i <= n_parameters; ++i) {
         expected_value *= i;
      }

      CHECK(std::abs(value - expected_value) < tolerance);
   }

   SECTION("Adds no contribution to gradient for parameters inside simplex")
   {
      const int n_parameters = 4;

      Local_linear_model model(n_parameters);
      std::vector<double> theta(n_parameters);
      theta[0] = 0.1;
      theta[1] = 0.05;
      theta[2] = 0;
      theta[3] = 0.4;
      model.set_parameters(theta);
      model.epsilon = 0.2;

      FEMH1BinLinear_parameters_uniform_prior prior;

      Eigen::VectorXd gradient(
         Eigen::VectorXd::Random(n_parameters));
      const Eigen::VectorXd initial_gradient(gradient);

      prior.add_log_gradient(model, gradient);

      for (int i = 0; i < n_parameters; ++i) {
         CHECK(gradient(i) == initial_gradient(i));
      }
   }

   SECTION("Analytical gradient matches numerical gradient")
   {
      const int n_parameters = 5;
      const double epsilon = 1.2;
      const double h = 1e-6;
      const double tolerance = 1e-10;

      Local_linear_model model(n_parameters);
      std::vector<double> theta(n_parameters);
      theta[0] = 0.1;
      theta[1] = 0.05;
      theta[2] = 0.1;
      theta[3] = 0.4;
      theta[4] = 0.3;
      model.set_parameters(theta);
      model.epsilon = epsilon;

      FEMH1BinLinear_parameters_uniform_prior prior;

      Eigen::VectorXd analytic_gradient(
         Eigen::VectorXd::Zero(n_parameters));

      prior.add_log_gradient(model, analytic_gradient);

      const auto log_value = [epsilon, &prior](const Eigen::VectorXd& x) {
         const int n_parameters = x.size();
         Local_linear_model model(n_parameters);
         std::vector<double> theta(n_parameters);
         for (int i = 0; i < n_parameters; ++i) {
            theta[i] = x(i);
         }
         model.set_parameters(theta);
         model.epsilon = epsilon;

         return prior.log_value(model);
      };

      Eigen::VectorXd parameters(n_parameters);
      for (int i = 0; i < n_parameters; ++i) {
         parameters(i) = theta[i];
      }

      Eigen::VectorXd numerical_gradient(n_parameters);
      for (int i = 0; i < n_parameters; ++i) {
         numerical_gradient(i) = central_difference_derivative(
            log_value, parameters, i, h);

         CHECK(std::abs(analytic_gradient(i) - numerical_gradient(i))
               < tolerance);
      }
   }
}

TEST_CASE("Test FEM-H1-BIN linear model exponential priors",
          "[femh1_bin_linear]")
{
   SECTION("Returns zero for parameters outside of simplex")
   {
      const int n_parameters = 6;

      Local_linear_model model(n_parameters);
      std::vector<double> theta(n_parameters, 1);
      model.set_parameters(theta);
      model.epsilon = 0;

      FEMH1BinLinear_parameters_exponential_prior prior;
      prior.alpha = 0.1;

      const double value = prior(model);
      const double expected_value = 0;

      CHECK(value == expected_value);
   }

   SECTION("Returns expected value for parameters inside simplex")
   {
      const int n_parameters = 4;
      const double tolerance = 1.0e-10;

      Local_linear_model model(n_parameters);
      std::vector<double> theta(n_parameters);
      theta[0] = 0.;
      theta[1] = 0.5;
      theta[2] = 0.;
      theta[3] = 0.4;
      model.set_parameters(theta);
      model.epsilon = 1.5;

      FEMH1BinLinear_parameters_exponential_prior prior;
      prior.alpha = 3.2;

      const double value = prior(model);
      const double theta_sum = std::accumulate(
         std::begin(theta), std::end(theta), 0.0);
      const double expected_value = std::exp(-prior.alpha * theta_sum);

      CHECK(std::abs(value - expected_value) < tolerance);
   }

   SECTION("Analytical gradient matches numerical gradient")
   {
      const int n_parameters = 3;
      const double epsilon = 1.2;
      const double h = 1e-6;
      const double tolerance = 1e-10;

      Local_linear_model model(n_parameters);
      std::vector<double> theta(n_parameters);
      theta[0] = 0.1;
      theta[1] = 0.05;
      theta[2] = 0.4;
      model.set_parameters(theta);
      model.epsilon = epsilon;

      FEMH1BinLinear_parameters_exponential_prior prior;
      prior.alpha = 1.2;

      Eigen::VectorXd analytic_gradient(
         Eigen::VectorXd::Zero(n_parameters));

      prior.add_log_gradient(model, analytic_gradient);

      const auto log_value = [epsilon, &prior](const Eigen::VectorXd& x) {
         const int n_parameters = x.size();
         Local_linear_model model(n_parameters);
         std::vector<double> theta(n_parameters);
         for (int i = 0; i < n_parameters; ++i) {
            theta[i] = x(i);
         }
         model.set_parameters(theta);
         model.epsilon = epsilon;

         return prior.log_value(model);
      };

      Eigen::VectorXd parameters(n_parameters);
      for (int i = 0; i < n_parameters; ++i) {
         parameters(i) = theta[i];
      }

      Eigen::VectorXd numerical_gradient(n_parameters);
      for (int i = 0; i < n_parameters; ++i) {
         numerical_gradient(i) = central_difference_derivative(
            log_value, parameters, i, h);

         CHECK(std::abs(analytic_gradient(i) - numerical_gradient(i))
               < tolerance);
      }
   }
}

TEST_CASE("Test FEM-H1-BIN linear model softmax normal priors",
          "[femh1_bin_linear]")
{
   SECTION("Analytical gradient matches numerical gradient")
   {
      const int n_components = 2;
      const int n_samples = 100;
      const double h = 1e-6;
      const double tolerance = 1e-6;

      Eigen::MatrixXd softmax_affiliations(
         Eigen::MatrixXd::Random(n_components, n_samples));

      FEMH1BinLinear_softmax_affiliations_normal_prior prior;
      prior.alpha = Eigen::VectorXd::Ones(n_components);
      prior.inverse_covariance = Eigen::MatrixXd::Identity(
         n_components, n_components);

      const int n_parameters = n_components * n_samples;
      Eigen::VectorXd analytic_gradient(
         Eigen::VectorXd::Zero(n_parameters));

      prior.add_log_gradient(softmax_affiliations, analytic_gradient);

      Eigen::VectorXd parameters(n_parameters);
      int index = 0;
      for (int t = 0; t < n_samples; ++t) {
         for (int i = 0; i < n_components; ++i) {
            parameters(index) = softmax_affiliations(i, t);
            ++index;
         }
      }

      const auto log_value = [n_components, n_samples, &prior](const Eigen::VectorXd& x) {
         Eigen::MatrixXd softmax_affiliations(n_components, n_samples);

         int index = 0;
         for (int t = 0; t < n_samples; ++t) {
            for (int i = 0; i < n_components; ++i) {
               softmax_affiliations(i, t) = x(index);
               ++index;
            }
         }

         return prior.log_value(softmax_affiliations);
      };

      Eigen::VectorXd numerical_gradient(n_parameters);
      for (int i = 0; i < n_parameters; ++i) {
         numerical_gradient(i) = central_difference_derivative(
            log_value, parameters, i, h);

         CHECK(std::abs(analytic_gradient(i) - numerical_gradient(i))
               < tolerance);
      }
   }
}

TEST_CASE("Test FEM-H1-BIN linear model log-likelihood",
          "[femh1_bin_linear]")
{
   SECTION("Analytical gradient matches numerical gradient with single component")
   {
      const int n_features = 4;
      const int n_components = 1;
      const int n_samples = 50;
      const double h = 1e-6;
      const double tolerance = 1e-6;

      Eigen::VectorXd outcomes(
         Eigen::VectorXd::Random(n_samples));
      Eigen::MatrixXd predictors(
         Eigen::MatrixXd::Random(n_features, n_samples));

      for (int t = 0; t < n_samples; ++t) {
         outcomes(t) = outcomes(t) > 0 ? 1 : 0;
         const double avg = predictors.col(t).sum() / n_features;
         for (int i = 0; i < n_features; ++i) {
            predictors(i, t) = predictors(i, t) - avg > 0 ? 1 : 0;
         }
      }

      const Eigen::MatrixXd softmax_affiliations(
         Eigen::MatrixXd::Zero(n_components, n_samples));

      const double epsilon = 0;
      Local_linear_model m1(n_features);
      m1.epsilon = epsilon;
      std::vector<double> theta1(n_features);
      theta1[0] = 0.15;
      theta1[1] = 0.2;
      theta1[2] = 0.5;
      m1.set_parameters(theta1);

      std::vector<Local_linear_model> models(n_components);
      models[0] = m1;

      const int n_parameters = n_components * n_features;

      Eigen::VectorXd analytic_gradient(
         Eigen::VectorXd::Zero(n_parameters));

      add_femh1_bin_linear_log_likelihood_gradient(
         outcomes, predictors, models, softmax_affiliations,
         analytic_gradient);

      Eigen::VectorXd parameters(n_parameters);
      int index = 0;
      for (int i = 0; i < n_components; ++i) {
         const auto theta = models[i].get_parameters();
         for (int j = 0; j < n_features; ++j) {
            parameters(index) = theta[j];
            ++index;
         }
      }

      const auto log_value = [n_features, n_components, n_samples, epsilon,
                              &outcomes, &predictors](
         const Eigen::VectorXd& x) {
         std::vector<Local_linear_model> models(n_components);
         Eigen::MatrixXd softmax_affiliations(
            Eigen::MatrixXd::Zero(n_components, n_samples));

         int index = 0;
         for (int i = 0; i < n_components; ++i) {
            Local_linear_model m(n_features);
            std::vector<double> theta(n_features);
            for (int j = 0; j < n_features; ++j) {
               theta[j] = x(index);
               ++index;
            }
            m.set_parameters(theta);
            m.epsilon = epsilon;
            models[i] = m;
         }

         return femh1_bin_linear_log_likelihood(
            outcomes, predictors, models, softmax_affiliations);
      };

      Eigen::VectorXd numerical_gradient(n_parameters);
      for (int i = 0; i < n_parameters; ++i) {
         numerical_gradient(i) = central_difference_derivative(
            log_value, parameters, i, h);

         CHECK(std::abs(analytic_gradient(i) - numerical_gradient(i))
               < tolerance);
      }
   }

   SECTION("Analytical gradient matches numerical gradient with multiple components")
   {
      const int n_features = 3;
      const int n_components = 3;
      const int n_samples = 40;
      const double h = 1e-6;
      const double tolerance = 1e-7;

      Eigen::VectorXd outcomes(
         Eigen::VectorXd::Random(n_samples));
      Eigen::MatrixXd predictors(
         Eigen::MatrixXd::Random(n_features, n_samples));

      for (int t = 0; t < n_samples; ++t) {
         outcomes(t) = outcomes(t) > 0 ? 1 : 0;
         const double avg = predictors.col(t).sum() / n_features;
         for (int i = 0; i < n_features; ++i) {
            predictors(i, t) = predictors(i, t) - avg > 0 ? 1 : 0;
         }
      }

      const Eigen::MatrixXd softmax_affiliations(
         Eigen::MatrixXd::Random(n_components, n_samples));

      const double epsilon = 0;
      Local_linear_model m1(n_features);
      m1.epsilon = epsilon;
      std::vector<double> theta1(n_features);
      theta1[0] = 0.1;
      theta1[1] = 0.3;
      theta1[2] = 0.44;
      m1.set_parameters(theta1);

      Local_linear_model m2(n_features);
      m2.epsilon = epsilon;
      std::vector<double> theta2(n_features);
      theta2[0] = 0.45;
      theta2[1] = 0.23;
      theta2[2] = 0.05;
      m2.set_parameters(theta2);

      Local_linear_model m3(n_features);
      m3.epsilon = epsilon;
      std::vector<double> theta3(n_features);
      theta3[0] = 0.1;
      theta3[1] = 0.1;
      theta3[2] = 0.1;
      m3.set_parameters(theta1);

      std::vector<Local_linear_model> models(n_components);
      models[0] = m1;
      models[1] = m2;
      models[2] = m3;

      const int n_parameters = n_components * (n_features + n_samples);

      Eigen::VectorXd analytic_gradient(
         Eigen::VectorXd::Zero(n_parameters));

      add_femh1_bin_linear_log_likelihood_gradient(
         outcomes, predictors, models, softmax_affiliations,
         analytic_gradient);

      Eigen::VectorXd parameters(n_parameters);
      int index = 0;
      for (int i = 0; i < n_components; ++i) {
         const auto theta = models[i].get_parameters();
         for (int j = 0; j < n_features; ++j) {
            parameters(index) = theta[j];
            ++index;
         }
      }
      for (int t = 0; t < n_samples; ++t) {
         for (int i = 0; i < n_components; ++i) {
            parameters(index) = softmax_affiliations(i, t);
            ++index;
         }
      }

      const auto log_value = [n_features, n_components, n_samples, epsilon,
                              &outcomes, &predictors](
         const Eigen::VectorXd& x) {
         std::vector<Local_linear_model> models(n_components);
         Eigen::MatrixXd softmax_affiliations(n_components, n_samples);

         int index = 0;
         for (int i = 0; i < n_components; ++i) {
            Local_linear_model m(n_features);
            std::vector<double> theta(n_features);
            for (int j = 0; j < n_features; ++j) {
               theta[j] = x(index);
               ++index;
            }
            m.set_parameters(theta);
            m.epsilon = epsilon;
            models[i] = m;
         }
         for (int t = 0; t < n_samples; ++t) {
            for (int i = 0; i < n_components; ++i) {
               softmax_affiliations(i, t) = x(index);
               ++index;
            }
         }

         return femh1_bin_linear_log_likelihood(
            outcomes, predictors, models, softmax_affiliations);
      };

      Eigen::VectorXd numerical_gradient(n_parameters);
      for (int i = 0; i < n_parameters; ++i) {
         numerical_gradient(i) = central_difference_derivative(
            log_value, parameters, i, h);

         CHECK(std::abs(analytic_gradient(i) - numerical_gradient(i))
               < tolerance);
      }
   }
}
