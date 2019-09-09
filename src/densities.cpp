#include "densities.hpp"

#include <Eigen/LU>

#include <cmath>

namespace pypbn {

Eigen::VectorXd softmax(const Eigen::VectorXd& x)
{
   Eigen::VectorXd result = x.array().exp().matrix();
   const double norm = result.sum();
   return result / norm;
}

double log_dirichlet_density(const Eigen::VectorXd& x,
                             const Eigen::VectorXd& alpha,
                             double tolerance)
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
   static const double pi = 4.0 * std::atan(1);

   const int n_dims = x.size();
   const double det_sigma_inverse = sigma_inverse.determinant();
   const double residual = (x - mu).dot(sigma_inverse * (x - mu));

   return -0.5 * n_dims * std::log(2. * pi) + 0.5 * std::log(det_sigma_inverse)
      - 0.5 * residual;
}

} // namespace pypbn
