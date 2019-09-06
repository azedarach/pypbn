#include "femh1_bin_linear_hmc.hpp"
#include "densities.hpp"

#include <cmath>

namespace pypbn {

namespace {

template <class Parameters_prior, class Softmax_affiliations_prior>
double log_target_density(const Eigen::VectorXd& outcomes,
                          const Eigen::MatrixXd& predictors,
                          const std::vector<Local_linear_model>& models,
                          const Eigen::MatrixXd& softmax_affiliations,
                          const Parameters_prior& parameters_prior,
                          const Softmax_affiliations_prior& affiliations_prior)
{
   double log_density =
      femh1_bin_linear_log_likelihood(outcomes, predictors, models,
                                      softmax_affiliations);

   const std::size_t n_components = models.size();
   for (std::size_t i = 0; i < n_components; ++i) {
      log_density += parameters_prior.log_value(models[i]);
   }

   if (n_components > 1) {
      log_density += affiliations_prior.log_value(softmax_affiliations);
   }

   return log_density;
}

template <class Parameters_prior, class Softmax_affiliations_prior>
double evaluate_energy(const Eigen::VectorXd& outcomes,
                       const Eigen::MatrixXd& predictors,
                       const std::vector<Local_linear_model>& models,
                       const Eigen::MatrixXd& softmax_affiliations,
                       const Parameters_prior& parameters_prior,
                       const Softmax_affiliations_prior& affiliations_prior)
{
   return -log_target_density(outcomes, predictors, models,
                              softmax_affiliations,
                              parameters_prior, affiliations_prior);
}

template <class Parameters_prior, class Softmax_affiliations_prior>
void fill_energy_gradient(const Eigen::VectorXd& outcomes,
                          const Eigen::MatrixXd& predictors,
                          const std::vector<Local_linear_model>& models,
                          const Eigen::MatrixXd& softmax_affiliations,
                          const Parameters_prior& parameters_prior,
                          const Softmax_affiliations_prior& affiliations_prior,
                          Eigen::VectorXd& gradient)
{
   const int n_components = softmax_affiliations.rows();
   const int n_samples = softmax_affiliations.cols();
   const int n_parameters = gradient.size();
   gradient = Eigen::VectorXd::Zero(n_parameters);

   add_femh1_bin_linear_log_likelihood_gradient(
      outcomes, predictors, models, softmax_affiliations, gradient);

   int gradient_index = 0;
   for (int i = 0; i < n_components; ++i) {
      const int n_parameters = models[i].get_number_of_parameters();
      auto gradient_component = gradient.segment(gradient_index, n_parameters);
      parameters_prior.add_log_gradient(models[i], gradient_component);
      gradient_index += n_parameters;
   }

   if (n_components > 1) {
      auto gradient_component = gradient.segment(
         gradient_index, n_components * n_samples);
      affiliations_prior.add_log_gradient(
         softmax_affiliations, gradient_component);
   }

   gradient *= -1;
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
   const Eigen::Ref<const Eigen::VectorXd> outcomes_,
   const Eigen::Ref<const Eigen::MatrixXd> predictors_,
   const Eigen::Ref<const Eigen::MatrixXd> parameters_,
   const Eigen::Ref<const Eigen::MatrixXd> affiliations_,
   double epsilon_theta_, double epsilon_gamma_,
   int n_leapfrog_steps_, double leapfrog_step_size_,
   int verbosity_, int random_seed_)
   : generator(random_seed_)
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

   parameters_prior.alpha = epsilon_theta_;
   softmax_affiliations_prior.alpha = Eigen::VectorXd::Ones(n_components);
   softmax_affiliations_prior.inverse_covariance = epsilon_gamma_
      * Eigen::MatrixXd::Identity(n_components, n_components);

   outcomes = outcomes_;
   predictors = predictors_;
   if (n_components > 1) {
      log_affiliations = affiliations_.array().log().matrix();
   } else {
      log_affiliations = Eigen::MatrixXd::Zero(n_components, n_samples);
   }

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

      current_energy = evaluate_energy(
         outcomes, predictors, models, log_affiliations,
         parameters_prior, softmax_affiliations_prior);

      fill_energy_gradient(outcomes, predictors, models,
                           log_affiliations, parameters_prior,
                           softmax_affiliations_prior,
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

      fill_energy_gradient(outcomes, predictors, temp_models,
                           temp_log_affiliations,
                           parameters_prior, softmax_affiliations_prior,
                           new_energy_gradient);

      momenta -= 0.5 * step_direction * leapfrog_step_size
         * new_energy_gradient;
   }

   const double new_energy = evaluate_energy(
      outcomes, predictors, temp_models, temp_log_affiliations,
      parameters_prior, softmax_affiliations_prior);
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

   Eigen::MatrixXd affiliations(
      Eigen::MatrixXd::Zero(n_components, n_samples));
   for (int t = 0; t < n_samples; ++t) {
      affiliations.col(t) = softmax(log_affiliations.col(t));
   }

   return affiliations;
}

double FEMH1BinLinearHMC::get_log_likelihood() const
{
   return femh1_bin_linear_log_likelihood(
      outcomes, predictors, models, log_affiliations);
}

} // namespace pypbn
