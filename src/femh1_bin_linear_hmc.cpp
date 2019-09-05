#include "femh1_bin_linear_hmc.hpp"

#include <Eigen/LU>

#include <cmath>

namespace pypbn {

namespace {

Eigen::VectorXd softmax(const Eigen::VectorXd& x)
{
   Eigen::VectorXd result = x.array().exp().matrix();
   const double norm = result.sum();
   return result / norm;
}

double inverse_simplex_volume(int np1)
{
   double fact = 1.;
   for (int i = 2; i <= np1; ++i) {
      fact *= i;
   }
   return fact;
}

double log_dirichlet_density(const Eigen::VectorXd& x,
                             const Eigen::VectorXd& alpha,
                             double tolerance = 1e-10)
{
   const int n_dims = x.size();

   double sum = 0.;
   for (int i = 0; i < n_dims; ++i) {
      if (x(i) <= 0 || x(i) >= 1) {
         return -std::numeric_limits<double>::max();
      }
      sum += x(i);
   }

   if (std::abs(sum - 1) > tolerance) {
      return -std::numeric_limits<double>::max();
   }

   double result = 0;

   for (int i = 0; i < n_dims; ++i) {
      result += ((alpha(i) - 1.) * std::log(x(i))
                 - std::lgamma(alpha(i)));
   }

   result += std::lgamma(alpha.sum());

   return result;
}

double log_normal_density(const Eigen::VectorXd& x, const Eigen::VectorXd& mu,
                          const Eigen::MatrixXd& sigma_inverse)
{
   constexpr double pi = 4.0 * std::atan(1);

   const int n_dims = x.size();
   const double det_sigma_inverse = sigma_inverse.determinant();
   const double residual = (x - mu).dot(sigma_inverse * (x - mu));

   return -0.5 * n_dims * std::log(2. * pi) + 0.5 * std::log(det_sigma_inverse)
      - 0.5 * residual;
}

double log_parameters_prior(const Local_linear_model& model)
{
   const std::vector<double>& parameters = model.get_parameters();
   const int n_parameters = parameters.size();

   double sum = 0.0;
   for (int i = 0; i < n_parameters; ++i) {
      if (parameters[i] < 0) {
         return -std::numeric_limits<double>::max();
      }
      sum += parameters[i];
   }
   if (sum <= 0 || sum >= 1) {
      return -std::numeric_limits<double>::max();
   }

   return inverse_simplex_volume(n_parameters);
}

void add_log_parameters_prior_gradient(const Local_linear_model& /* model */,
                                       Eigen::VectorXd& /* gradient */)
{
   // trivial for uniform prior
}

double log_affiliations_prior(const Eigen::MatrixXd& log_affiliations,
                              const Eigen::MatrixXd& sigma_inverse)
{
   // Dirichlet prior for t = 1 combined with conditional
   // log-normal distributions for t = 2, ..., T
   const int n_samples = log_affiliations.cols();
   const int n_components = log_affiliations.rows();

   const Eigen::VectorXd alpha(Eigen::VectorXd::Ones(n_components));
   const Eigen::VectorXd gamma(softmax(log_affiliations.col(0)));
   double result = log_dirichlet_density(gamma, alpha);

   for (int t = 1; t < n_samples; ++t) {
      result += log_normal_density(log_affiliations.col(t),
                                   log_affiliations.col(t - 1),
                                   sigma_inverse);
   }

   return result;
}

void add_log_affiliations_prior_gradient(
   const Eigen::MatrixXd& log_affiliations,
   const Eigen::MatrixXd& sigma_inverse,
   Eigen::VectorXd& gradient)
{
   const int n_components = log_affiliations.rows();

   // for n_components == 1 the affiliations sequence is
   // fully constrained
   if (n_components > 1) {
      const int n_samples = log_affiliations.cols();

      const int n_parameters = gradient.size();

      int gradient_index = n_parameters - n_components * n_samples;

      // NB assumes prior for initial values is uniform and
      // independent of values, so only contribution to gradient
      // comes from conditional densities for t = 2, ..., T
      for (int t = 0; t < n_samples; ++t) {
         if (t < n_samples - 1) {
            const Eigen::VectorXd g = sigma_inverse * (
               log_affiliations.col(t + 1) - log_affiliations.col(t));
            gradient.segment(gradient_index, n_components) += g;
         }
         if (t > 0) {
            const Eigen::VectorXd g = sigma_inverse * (
               log_affiliations.col(t) - log_affiliations.col(t - 1));
            gradient.segment(gradient_index, n_components) -= g;
         }
         gradient_index += n_components;
      }
   }
}

double log_likelihood(const Eigen::VectorXd& outcomes,
                      const Eigen::MatrixXd& predictors,
                      const std::vector<Local_linear_model>& models,
                      const Eigen::MatrixXd& log_affiliations)
{
   double log_like = 0;

   const int n_features = predictors.rows();
   const int n_samples = log_affiliations.cols();
   const int n_components = models.size();

   for (int t = 0; t < n_samples; ++t) {
      Eigen::VectorXd gamma(softmax(log_affiliations.col(t)));

      Eigen::VectorXd theta(Eigen::VectorXd::Zero(n_features));
      for (int i = 0; i < n_components; ++i) {
         const auto& predictor_indices = models[i].get_predictor_indices();
         const auto& parameters = models[i].get_parameters();
         const int n_parameters = parameters.size();
         for (int j = 0; j < n_parameters; ++j) {
            theta(predictor_indices[j]) += gamma(i) * parameters[j];
         }
      }

      const double p = theta.dot(predictors.col(t));

      log_like += outcomes(t) * std::log(p)
         + (1 - outcomes(t)) * std::log(1 - p);
   }

   return log_like;
}

void add_log_likelihood_gradient(
   const Eigen::VectorXd& outcomes, const Eigen::MatrixXd& predictors,
   const std::vector<Local_linear_model>& models,
   const Eigen::MatrixXd& log_affiliations, Eigen::VectorXd& gradient)
{
   const int n_features = predictors.rows();
   const int n_components = models.size();
   const int n_samples = log_affiliations.cols();

   Eigen::MatrixXd local_theta(
      Eigen::VectorXd::Zero(n_features, n_components));
   for (int i = 0; i < n_components; ++i) {
      const auto& predictor_indices = models[i].get_predictor_indices();
      const auto& parameters = models[i].get_parameters();
      const int n_parameters = parameters.size();
      for (int j = 0; j < n_parameters; ++j) {
         local_theta(predictor_indices[j], i) = parameters[j];
      }
   }

   for (int t = 0; t < n_samples; ++t) {
      Eigen::VectorXd gamma(softmax(log_affiliations.col(t)));

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

double penalised_log_likelihood(const Eigen::VectorXd& outcomes,
                                const Eigen::MatrixXd& predictors,
                                const std::vector<Local_linear_model>& models,
                                const Eigen::MatrixXd& log_affiliations)
{
   double log_like = log_likelihood(outcomes, predictors, models,
                                    log_affiliations);

   const int n_components = models.size();
   for (int i = 0; i < n_components; ++i) {
      log_like -= models[i].regularization();
   }

   return log_like;
}

void add_penalised_log_likelihood_gradient(
   const Eigen::VectorXd& outcomes, const Eigen::MatrixXd& predictors,
   const std::vector<Local_linear_model>& models,
   const Eigen::MatrixXd& log_affiliations,
   Eigen::VectorXd& gradient)
{
   add_log_likelihood_gradient(
      outcomes, predictors, models, log_affiliations, gradient);

   const int n_components = models.size();
   int gradient_index = 0;
   for (int i = 0; i < n_components; ++i) {
      const int n_parameters = models[i].get_number_of_parameters();
      for (int j = 0; j < n_parameters; ++j) {
         gradient(gradient_index) -= models[i].regularization_gradient(j);
         ++gradient_index;
      }
   }
}

double log_target_density(const Eigen::VectorXd& outcomes,
                          const Eigen::MatrixXd& predictors,
                          const std::vector<Local_linear_model>& models,
                          const Eigen::MatrixXd& log_affiliations,
                          const Eigen::MatrixXd& sigma_inverse)
{
   double log_density =
      penalised_log_likelihood(outcomes, predictors, models, log_affiliations);

   const std::size_t n_components = models.size();
   for (std::size_t i = 0; i < n_components; ++i) {
      log_density += log_parameters_prior(models[i]);
   }

   if (n_components > 1) {
      log_density += log_affiliations_prior(log_affiliations, sigma_inverse);
   }

   return log_density;
}

void gradient_log_target_density(const Eigen::VectorXd& outcomes,
                                 const Eigen::MatrixXd& predictors,
                                 const std::vector<Local_linear_model>& models,
                                 const Eigen::MatrixXd& log_affiliations,
                                 const Eigen::MatrixXd& sigma_inverse,
                                 Eigen::VectorXd& gradient)
{
   const int n_parameters = gradient.size();
   gradient = Eigen::VectorXd::Zero(n_parameters);

   add_penalised_log_likelihood_gradient(
      outcomes, predictors, models, log_affiliations, gradient);

   const std::size_t n_components = models.size();
   for (std::size_t i = 0; i < n_components; ++i) {
      add_log_parameters_prior_gradient(models[i], gradient);
   }

   if (n_components > 1) {
      add_log_affiliations_prior_gradient(log_affiliations, sigma_inverse,
                                          gradient);
   }
}

void accept_positions(
   const Eigen::VectorXd& positions, std::vector<Local_linear_model>& models,
   Eigen::MatrixXd& log_affiliations)
{
   const int n_components = models.size();
   const int n_samples = log_affiliations.cols();

   int position_index = 0;
   for (int i = 0; i < n_components; ++i) {
      const int n_parameters = models[i].get_number_of_parameters();
      std::vector<double> theta(n_parameters, 0);
      for (int j = 0; j < n_parameters; ++j) {
         theta[j] = positions(position_index);
         ++position_index;
      }
      models[i].set_parameters(theta);
   }

   if (n_components > 1) {
      for (int t = 0; t < n_samples; ++t) {
         for (int i = 0; i < n_components; ++i) {
            log_affiliations(i, t) = positions(position_index);
            ++position_index;
         }
      }
   }
}

template <class Generator>
void initialize_momenta(Eigen::VectorXd& momenta, Generator& generator)
{
   std::normal_distribution<> dist(0., 1.);

   const int n_momenta = momenta.size();
   for (int i = 0; i < n_momenta; ++i) {
      momenta(i) = dist(generator);
   }
}

void initialize_positions(
   const std::vector<Local_linear_model>& models,
   const Eigen::MatrixXd& log_affiliations,
   Eigen::VectorXd& positions)
{
   const int n_components = models.size();
   const int n_samples = log_affiliations.cols();

   int n_parameters = 0;
   for (int i = 0; i < n_components; ++i) {
      n_parameters += models[i].get_number_of_parameters();
   }

   if (n_components > 1) {
      n_parameters += n_components * n_samples;
   }

   if (positions.size() != n_parameters) {
      throw std::runtime_error(
         "number of parameters does not match size of positions vector");
   }

   int position_index = 0;
   for (int i = 0; i < n_components; ++i) {
      const std::vector<double>& theta = models[i].get_parameters();
      for (const auto& p : theta) {
         positions(position_index) = p;
         ++position_index;
      }
   }

   if (n_components > 1) {
      for (int t = 0; t < n_samples; ++t) {
         for (int i = 0; i < n_components; ++i) {
            positions(position_index) = log_affiliations(i, t);
            ++position_index;
         }
      }
   }
}

} // anonymous namespace

FEMH1BinLinearHMC::FEMH1BinLinearHMC(
   const Eigen::Ref<const Eigen::MatrixXd>& outcomes_,
   const Eigen::Ref<const Eigen::MatrixXd>& predictors_,
   const Eigen::Ref<const Eigen::MatrixXd>& parameters_,
   const Eigen::Ref<const Eigen::MatrixXd>& affiliations_,
   double epsilon_theta_, double epsilon_gamma_,
   int n_leapfrog_steps_, double leapfrog_step_size_,
   int verbosity_, int random_seed_)
   : generator(random_seed_)
   , outcomes(outcomes_)
   , predictors(predictors_)
   , verbosity(verbosity_)
   , n_leapfrog_steps(n_leapfrog_steps_)
   , leapfrog_step_size(leapfrog_step_size_)
   , chain_length(0)
   , acceptance_rate(0)
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

   if (n_components > 1) {
      log_affiliations = affiliations_.array().log().matrix();
   } else {
      log_affiliations = Eigen::MatrixXd::Zero(n_components, n_samples);
   }

   sigma_inverse = epsilon_gamma_ * Eigen::MatrixXd::Identity(
      n_components, n_components);

   models = std::vector<Local_linear_model>(n_components);
   for (int i = 0; i < n_components; ++i) {
      models[i] = Local_linear_model(n_features);
      std::vector<double> theta(n_features);
      for (int j = 0; j < n_features; ++j) {
         theta[j] = parameters_(i, j);
      }
      models[i].set_parameters(theta);
      models[i].epsilon = epsilon_theta_;
   }

   temp_models = models;
   temp_log_affiliations = log_affiliations;

   const int n_parameters = (n_components == 1 ?
                             n_components * n_features :
                             n_components * (n_features + n_samples));

   positions = Eigen::VectorXd::Zero(n_parameters);
   momenta = Eigen::VectorXd::Zero(n_parameters);
   current_energy_gradient = Eigen::VectorXd::Zero(n_parameters);
   new_energy_gradient = Eigen::VectorXd::Zero(n_parameters);
}

bool FEMH1BinLinearHMC::hmc_step()
{
   if (chain_length == 0) {
      initialize_positions(models, log_affiliations, positions);

      current_energy = log_target_density(
         outcomes, predictors, models, log_affiliations, sigma_inverse);
      gradient_log_target_density(outcomes, predictors, models,
                                  log_affiliations, sigma_inverse,
                                  current_energy_gradient);
   }

   initialize_momenta(momenta, generator);

   const double current_H = current_energy + 0.5 * momenta.squaredNorm();

   std::uniform_real_distribution<> uniform_dist(0., 1.);
   const double direction_choice = uniform_dist(generator);
   const double step_direction = (direction_choice < 0.5 ? 1 : -1);

   new_energy_gradient = current_energy_gradient;
   for (int tau = 0; tau < n_leapfrog_steps; ++tau) {
      momenta -= 0.5 * step_direction * leapfrog_step_size
         * new_energy_gradient;

      positions += step_direction * leapfrog_step_size * momenta;

      accept_positions(positions, temp_models, temp_log_affiliations);

      gradient_log_target_density(outcomes, predictors, temp_models,
                                  temp_log_affiliations, sigma_inverse,
                                  new_energy_gradient);

      momenta -= 0.5 * step_direction * leapfrog_step_size
         * new_energy_gradient;
   }

   const double new_energy = log_target_density(
      outcomes, predictors, temp_models,
      temp_log_affiliations, sigma_inverse);
   const double new_H = new_energy + 0.5 * momenta.squaredNorm();
   const double delta_H = new_H - current_H;

   bool accept = false;
   if (delta_H < 0) {
      accept = true;
   } else {
      const double p = uniform_dist(generator);
      if (p < std::exp(-delta_H)) {
         accept = true;
      } else {
         accept = false;
      }
   }

   const double n_accepted = acceptance_rate * chain_length;
   if (accept) {
      accept_positions(positions, models, log_affiliations);

      current_energy = new_energy;
      current_energy_gradient = new_energy_gradient;

      acceptance_rate = (1 + n_accepted) / (chain_length + 1);
   } else {
      initialize_positions(models, log_affiliations, positions);

      acceptance_rate = n_accepted / (chain_length + 1);
   }

   ++chain_length;

   return true;
}

void FEMH1BinLinearHMC::reset()
{
   chain_length = 0;
   acceptance_rate = 0;
}

Eigen::MatrixXd FEMH1BinLinearHMC::get_parameters() const
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

Eigen::MatrixXd FEMH1BinLinearHMC::get_affiliations() const
{
   const int n_components = log_affiliations.rows();
   const int n_samples = log_affiliations.cols();

   Eigen::MatrixXd affiliations(n_components, n_samples);
   for (int t = 0; t < n_samples; ++t) {
      affiliations.col(t) = softmax(log_affiliations.col(t));
   }

   return affiliations;
}

double FEMH1BinLinearHMC::get_log_likelihood() const
{
   return log_likelihood(outcomes, predictors, models, log_affiliations);
}

} // namespace pypbn
