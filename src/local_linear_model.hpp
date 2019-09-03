#ifndef PYPBN_LOCAL_LINEAR_MODEL_HPP_INCLUDED
#define PYPBN_LOCAL_LINEAR_MODEL_HPP_INCLUDED

#include <Eigen/Core>

#include <numeric>
#include <vector>

namespace pypbn {

class Local_linear_model {
public:
   using Predictor_indices = std::vector<int>;
   using Parameters = std::vector<double>;

   double epsilon{0};

   Local_linear_model() = default;
   explicit Local_linear_model(int);
   explicit Local_linear_model(const Predictor_indices&);
   Local_linear_model(const Predictor_indices&,
                      const Parameters&);
   ~Local_linear_model() = default;
   Local_linear_model(const Local_linear_model&) = default;
   Local_linear_model(Local_linear_model&&) = default;
   Local_linear_model& operator=(const Local_linear_model&) = default;
   Local_linear_model& operator=(Local_linear_model&&) = default;

   int get_number_of_parameters() const;
   const Predictor_indices& get_predictor_indices() const { return predictor_indices; }
   const Parameters& get_parameters() const { return theta; }

   void set_parameter(int, double);
   void set_parameters(const Parameters&);

   double log_likelihood_component(double, const Eigen::VectorXd&) const;
   double score_component(int, double, const Eigen::VectorXd&) const;

   double regularization() const {
      return epsilon * std::accumulate(std::begin(theta), std::end(theta), 0.0);
   }
   double regularization_gradient(int) const {
      return epsilon;
   }

private:
   Predictor_indices predictor_indices{};
   Parameters theta{};
};

} // namespace pypbn

#endif