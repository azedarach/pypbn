#ifndef PYPBN_DENSITIES_HPP_INCLUDED
#define PYPBN_DENSITIES_HPP_INCLUDED

#include <Eigen/Core>

#include <limits>

namespace pypbn {

Eigen::VectorXd softmax(const Eigen::VectorXd&);

double log_dirichlet_density(const Eigen::VectorXd&, const Eigen::VectorXd&,
                             double tolerance = std::numeric_limits<double>::epsilon());
double log_normal_density(const Eigen::VectorXd&, const Eigen::VectorXd&,
                          const Eigen::MatrixXd&);

} // namespace pypbn

#endif
