#include "local_linear_model.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pypbn {

Local_linear_model::Local_linear_model(int n_features)
{
   if (n_features < 1) {
      throw std::runtime_error(
         "number of features must be at least one");
   }

   theta = std::vector<double>(n_features, 0);
   predictor_indices = std::vector<int>(n_features);
   std::iota(std::begin(predictor_indices), std::end(predictor_indices), 0);
}

Local_linear_model::Local_linear_model(
   const Predictor_indices& predictor_indices_)
   : predictor_indices(predictor_indices_)
{
   const auto n_features = predictor_indices_.size();

   if (n_features < 1) {
      throw std::runtime_error(
         "number of features must be at least one");
   }

   theta = std::vector<double>(n_features, 0);
}

Local_linear_model::Local_linear_model(
   const Predictor_indices& predictor_indices_,
   const Parameters& theta_)
   : predictor_indices(predictor_indices_)
   , theta(theta_)
{
   const int n_features = predictor_indices_.size();

   if (n_features < 1) {
      throw std::runtime_error(
         "number of features must be at least one");
   }

   const int n_pars = theta_.size();
   if (n_features != n_pars) {
      throw std::runtime_error(
         "number of predictor indices does not match "
         "number of parameters");
   }
}

int Local_linear_model::get_number_of_parameters() const
{
   return theta.size();
}

void Local_linear_model::set_parameter(int i, double p)
{
   const int n_features = theta.size();
   if (i < 0 || i >= n_features) {
      throw std::runtime_error("invalid parameter index");
   }

   theta[i] = p;
}

void Local_linear_model::set_parameters(const Parameters& theta_)
{
   if (theta.size() != theta_.size()) {
      throw std::runtime_error(
         "number of new parameters does not match number "
         "of old parameters");
   }

   theta = theta_;
}

double Local_linear_model::log_likelihood_component(double y, const Eigen::VectorXd& X) const
{
   using std::log;

   const int n_features = theta.size();
   double lp = 0;
   for (int i = 0; i < n_features; ++i) {
      lp += theta[i] * X(predictor_indices[i]);
   }

   if (lp <= 0) {
      if (y == 0) {
         return 0;
      } else {
         return y > 0 ? -std::numeric_limits<double>::max() :
            std::numeric_limits<double>::max();
      }
   }
   if (lp <= 0 || lp >= 1) {
      return -std::numeric_limits<double>::max();
   } else {
      return y * log(lp) + (1 - y) * log(1 - lp);
   }
}

double Local_linear_model::score_component(int parameter, double y, const Eigen::VectorXd& X) const
{
   using std::log;

   const int n_features = theta.size();
   if (parameter >= n_features) {
      throw std::runtime_error(
         "parameter index out of bounds");
   }

   double lp = 0;
   for (int i = 0; i < n_features; ++i) {
      lp += theta[i] * X(predictor_indices[i]);
   }

   const int idx = predictor_indices[parameter];
   return y * X(idx) / lp - (1 - y) * X(idx) / (1 - lp);
}

} // namespace pypbn
