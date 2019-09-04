#include "femh1_bin_linear_mc.hpp"

#include <Eigen/LU>

#include <cmath>

namespace pypbn {

namespace {

bool update_femh1_bin_parameters(
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
         const auto& parameters = models[i].get_parameters();
         const int n_parameters = parameters.size();
         for (int j = 0; j < n_parameters; ++j) {
            theta(j) += gamma(i) * parameters[j];
         }
      }

      const double p = theta.dot(predictors.col(t));

      log_like += outcomes(t) * std::log(p)
         + (1 - outcomes(t)) * std::log(1 - p);
   }

   return log_like;
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

   log_density += log_affiliations_prior(log_affiliations, sigma_inverse);

   return log_density;
}

// initial simple approach - independent normal densities for each
// parameter, expect it to be slow
template <class Generator>
std::tuple<std::vector<double>, double> get_trial_parameters(
   const std::vector<double>& theta, double sigma, Generator& generator)
{
   const int n_parameters = theta.size();

   std::normal_distribution<> dist(0., sigma);
   std::vector<double> theta_proposal(theta);

   Eigen::VectorXd theta_old(n_parameters);
   Eigen::VectorXd theta_new(n_parameters);
   for (int i = 0; i < n_parameters; ++i) {
      theta_proposal[i] = theta[i] + dist(generator);
      theta_old(i) = theta[i];
      theta_new(i) = theta[i];
   }

   Eigen::MatrixXd sigma_inverse(
      (1. / sigma) * Eigen::MatrixXd::Identity(n_parameters, n_parameters));

   const double log_q_forward = log_normal_density(
      theta_new, theta_old, sigma_inverse);
   const double log_q_backward = log_normal_density(
      theta_old, theta_new, sigma_inverse);
   const double delta_log_q = log_q_backward - log_q_forward;

   return std::make_tuple(theta_proposal, delta_log_q);
}

template <class Generator>
std::tuple<Eigen::MatrixXd, double> get_trial_log_affiliations(
   const Eigen::MatrixXd& log_affiliations, double sigma, Generator& generator)
{
   const int n_components = log_affiliations.rows();
   const int n_samples = log_affiliations.cols();

   std::normal_distribution<> dist(0., sigma);
   Eigen::MatrixXd perturbation(Eigen::MatrixXd::Zero(n_components, n_samples));
   for (int t = 0; t < n_samples; ++t) {
      for (int i = 0; i < n_components; ++i) {
         perturbation(i, t) = dist(generator);
      }
   }

   const Eigen::MatrixXd proposal =
      log_affiliations + perturbation;

   Eigen::VectorXd a_new(n_components * n_samples);
   Eigen::VectorXd a_old(n_components * n_samples);

   for (int i = 0; i < n_components; ++i) {
      a_old.segment(i * n_samples, n_samples) = log_affiliations.row(i);
      a_new.segment(i * n_samples, n_samples) = proposal.row(i);
   }

   const Eigen::MatrixXd sigma_inverse(
      (1. / sigma) * Eigen::MatrixXd::Identity(
         n_components * n_samples, n_components * n_samples));

   const double log_q_forward = log_normal_density(
      a_new, a_old, sigma_inverse);
   const double log_q_backward = log_normal_density(
      a_old, a_new, sigma_inverse);
   const double delta_log_q = 0; //log_q_backward - log_q_forward;

   return std::make_tuple(proposal, delta_log_q);
}

template <class Generator>
bool check_acceptance(double log_acceptance_prob, Generator& generator)
{
   if (log_acceptance_prob >= 0) {
      return true;
   }

   std::uniform_real_distribution<> dist(0., 1.);

   const double acceptance_prob = std::exp(log_acceptance_prob);
   const double u = dist(generator);

   return u <= acceptance_prob;
}

} // anonymous namespace

FEMH1BinLinearMC::FEMH1BinLinearMC(
   const Eigen::Ref<const Eigen::MatrixXd>& outcomes_,
   const Eigen::Ref<const Eigen::MatrixXd>& predictors_,
   const Eigen::Ref<const Eigen::MatrixXd>& parameters_,
   const Eigen::Ref<const Eigen::MatrixXd>& affiliations_,
   double epsilon_theta_, double epsilon_gamma_,
   double sigma_theta_, double sigma_gamma_,
   bool include_parameters_, double parameters_tolerance,
   Ipopt_initial_guess parameters_initialization,
   int max_parameters_iterations,
   int verbosity, int random_seed)
   : theta_solver(random_seed)
   , generator(random_seed)
   , include_parameters(include_parameters_)
   , sigma_theta(sigma_theta_)
   , sigma_gamma(sigma_gamma_)
   , chain_length(0)
   , affiliations_acceptance_rate(0)
   , model_acceptance_rates(std::vector<double>(parameters_.rows(), 0))
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

   theta_solver.set_tolerance(parameters_tolerance);
   theta_solver.set_max_iterations(max_parameters_iterations);
   theta_solver.set_verbosity(verbosity);
   theta_solver.set_initialization_method(parameters_initialization);
   theta_solver.initialize();

   outcomes = outcomes_;
   predictors = predictors_;
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
}

bool FEMH1BinLinearMC::metropolis_step()
{
   const int n_components = models.size();
   bool success = true;

   double current_log_density = log_target_density(
      outcomes, predictors, models, log_affiliations, sigma_inverse);

   if (include_parameters) {
      for (int i = 0; i < n_components; ++i) {
         const std::vector<double>& old_parameters = models[i].get_parameters();

         const std::tuple<std::vector<double>, double> proposal =
            get_trial_parameters(old_parameters, sigma_theta, generator);

         const std::vector<double>& trial_parameters = std::get<0>(proposal);
         const double delta_log_q = std::get<1>(proposal);

         models[i].set_parameters(trial_parameters);

         const double next_log_density = log_target_density(
            outcomes, predictors, models, log_affiliations, sigma_inverse);

         const double log_acceptance = next_log_density - current_log_density
            + delta_log_q;

         const double n_accepted = model_acceptance_rates[i] * chain_length;
         const bool accept = check_acceptance(log_acceptance, generator);
         if (accept) {
            model_acceptance_rates[i] = (1 + n_accepted) / (chain_length + 1);
            current_log_density = next_log_density;
         } else {
            model_acceptance_rates[i] = n_accepted / (chain_length + 1);
            models[i].set_parameters(old_parameters);
         }
      }
   } else {
      success = update_parameters();

      for (int i = 0; i < n_components; ++i) {
         model_acceptance_rates[i] = 1;
      }

      current_log_density = log_target_density(
         outcomes, predictors, models, log_affiliations, sigma_inverse);
   }

   if (n_components > 1) {
      const std::tuple<Eigen::MatrixXd, double> log_affiliations_proposal =
         get_trial_log_affiliations(log_affiliations, sigma_gamma, generator);

      const Eigen::MatrixXd& trial_log_affiliations = std::get<0>(
         log_affiliations_proposal);
      const double delta_log_q = std::get<1>(log_affiliations_proposal);

      const double next_log_density = log_target_density(
         outcomes, predictors, models, trial_log_affiliations, sigma_inverse);

      const double log_acceptance = next_log_density - current_log_density
         + delta_log_q;

      const bool affiliations_accepted = check_acceptance(
         log_acceptance, generator);
      const double n_accepted = affiliations_acceptance_rate * chain_length;
      if (affiliations_accepted) {
         current_log_density = next_log_density;
         log_affiliations = trial_log_affiliations;
         affiliations_acceptance_rate = (1 + n_accepted) / (chain_length + 1);
      } else {
         affiliations_acceptance_rate = n_accepted / (chain_length + 1);
      }
   } else {
      affiliations_acceptance_rate = 1;
   }

   ++chain_length;

   return success;
}

void FEMH1BinLinearMC::reset()
{
   chain_length = 0;
   affiliations_acceptance_rate = 0;

   const int n_components = model_acceptance_rates.size();
   for (int i = 0; i < n_components; ++i) {
      model_acceptance_rates[i] = 0;
   }
}

Eigen::MatrixXd FEMH1BinLinearMC::get_parameters() const
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

Eigen::MatrixXd FEMH1BinLinearMC::get_affiliations() const
{
   const int n_components = log_affiliations.rows();
   const int n_samples = log_affiliations.cols();

   Eigen::MatrixXd affiliations(n_components, n_samples);
   for (int t = 0; t < n_samples; ++t) {
      affiliations.col(t) = softmax(log_affiliations.col(t));
   }

   return affiliations;
}

double FEMH1BinLinearMC::get_log_likelihood() const
{
   return log_likelihood(outcomes, predictors, models, log_affiliations);
}

double FEMH1BinLinearMC::get_model_acceptance_rate(int i) const
{
   const int n_components = models.size();

   if (i < 0 || i >= n_components) {
      throw std::runtime_error("invalid model index");
   }

   return model_acceptance_rates[i];
}

bool FEMH1BinLinearMC::update_parameters()
{
   const Eigen::MatrixXd affiliations(get_affiliations());

   return update_femh1_bin_parameters(
      outcomes, predictors, affiliations, models, theta_solver);
}

} // namespace pypbn
