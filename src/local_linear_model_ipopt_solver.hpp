#ifndef PYPBN_LOCAL_LINEAR_MODEL_IPOPT_SOLVER_HPP_INCLUDED
#define PYPBN_LOCAL_LINEAR_MODEL_IPOPT_SOLVER_HPP_INCLUDED

#include "local_linear_model.hpp"

#include <Eigen/Core>

#include <IpIpoptApplication.hpp>
#include <IpTNLP.hpp>

#include <random>

namespace pypbn {

enum class Ipopt_initial_guess : int { Uniform, Random, Current };

class Local_linear_model_ipopt_solver {
public:
   Local_linear_model_ipopt_solver();
   explicit Local_linear_model_ipopt_solver(int);

   void set_initialization_method(Ipopt_initial_guess i) { initialization = i; }
   void set_max_iterations(int it);
   void set_n_trials(int n) { n_trials = n; }
   void set_tolerance(double t);
   void set_verbosity(int v) { verbosity = v; }

   void initialize();

   bool update_local_model(const Eigen::VectorXd&, const Eigen::MatrixXd&,
                           const Eigen::VectorXd&, Local_linear_model&);

private:
   Ipopt_initial_guess initialization{Ipopt_initial_guess::Uniform};
   int verbosity{0};
   int n_trials{10};
   std::mt19937 generator{};
   Ipopt::SmartPtr<Ipopt::IpoptApplication> ip_solver{};
};

} // namespace pypbn

#endif