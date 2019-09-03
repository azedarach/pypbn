#ifndef PYPBN_CLPSIMPLEX_AFFILIATIONS_SOLVER_HPP_INCLUDED
#define PYPBN_CLPSIMPLEX_AFFILIATIONS_SOLVER_HPP_INCLUDED

/**
 * @file clpsimplex_affiliations_solver.hpp
 * @brief contains definition of ClpSimplex_affiliations_solver class
 */

#include "linear_constraint.hpp"

#include <ClpSimplex.hpp>

#include <cstddef>
#include <stdexcept>
#include <memory>
#include <vector>

class CoinPackedVector;

namespace pypbn {

namespace detail {

template <class Matrix>
std::vector<double> stack_columns(const Matrix& A)
{
   const int n_rows = A.rows();
   const int n_cols = A.cols();

   std::vector<double> result(n_rows * n_cols);
   for (int j = 0; j < n_cols; ++j) {
      for (int i = 0; i < n_rows; ++i) {
         result[i + j * n_rows] = A(i, j);
      }
   }
   return result;
}

} // namespace detail

/**
 * @class ClpSimplex_affiliations_solver
 * @brief solves linear program to update affiliations
 */
class ClpSimplex_affiliations_solver {
public:
   using Index_type = int;

   enum class Status : int { SUCCESS = 0, FAIL };

   // Constant for representing unbounded variables
   static const double Infinity;

   template <class DistanceMatrix, class BasisMatrix>
   ClpSimplex_affiliations_solver(
      const DistanceMatrix&, const BasisMatrix&, double);
   ~ClpSimplex_affiliations_solver() = default;
   ClpSimplex_affiliations_solver(const ClpSimplex_affiliations_solver&) = delete;
   ClpSimplex_affiliations_solver(ClpSimplex_affiliations_solver&&) = default;
   ClpSimplex_affiliations_solver& operator=(const ClpSimplex_affiliations_solver&) = delete;
   ClpSimplex_affiliations_solver& operator=(ClpSimplex_affiliations_solver&&) = default;

   double get_max_tv_norm() const { return max_tv_norm; }
   Index_type get_n_components() const { return n_components; }
   Index_type get_n_elements() const { return n_elements; }
   Index_type get_n_samples() const { return n_samples; }

   void set_max_iterations(int it) { max_iterations = it; }
   int get_max_iterations() const { return max_iterations; }
   void set_verbosity(int v) { verbosity = v; }

   Index_type get_n_primary_variables() const;
   Index_type get_n_auxiliary_variables() const;
   Index_type get_n_total_variables() const { return solver->getNumCols(); }
   Index_type get_n_constraints() const { return solver->getNumRows(); }

   std::vector<double> get_objective_coefficients() const;
   template <class Matrix>
   void get_objective_coefficients(Matrix&) const;

   template <class DistanceMatrix>
   Status update_affiliations(const DistanceMatrix&);
   template <class Matrix>
   void get_affiliations(Matrix&) const;

private:
   static const double Minimize;
   static const double Maximize;

   Index_type n_components{0};
   Index_type n_elements{0};
   Index_type n_samples{0};
   double max_tv_norm{-1};
   int max_iterations{5000};
   int verbosity{0};

   std::vector<double> basis_values{};
   std::vector<Linear_constraint> equality_constraints{};
   std::vector<Linear_constraint> positivity_constraints{};
   std::vector<Linear_constraint> aux_positivity_constraints{};
   std::vector<Linear_constraint> aux_norm_constraints{};
   std::unique_ptr<ClpSimplex> solver{nullptr};

   void initialize_constraints_and_bounds();
   void initialize_equality_constraints();
   void initialize_basic_inequality_constraints();
   void initialize_auxiliary_inequality_constraints();
   void add_constraints();

   template <class DistanceMatrix>
   void update_objective(const DistanceMatrix&);
};

template <class DistanceMatrix, class BasisMatrix>
ClpSimplex_affiliations_solver::ClpSimplex_affiliations_solver(
   const DistanceMatrix& G, const BasisMatrix& V, double max_tv_norm_)
   : n_components(G.rows()), n_elements(V.rows())
   , n_samples(G.cols()), max_tv_norm(max_tv_norm_)
{
   basis_values = detail::stack_columns(V);

   solver = std::unique_ptr<ClpSimplex>(new ClpSimplex());
   solver->setOptimizationDirection(ClpSimplex_affiliations_solver::Minimize);

   if (max_tv_norm_ < 0) {
      solver->resize(0, n_components * n_elements);
   } else {
      solver->resize(0, 2 * n_components * n_elements);
   }

   if (n_components > 1) {
      initialize_constraints_and_bounds();
   }
   update_objective(G);
}

template <class DistanceMatrix>
void ClpSimplex_affiliations_solver::update_objective(
   const DistanceMatrix& G)
{
   const auto n_variables = n_elements * n_components;
   std::vector<double> c(n_variables, 0);

   for (Index_type t = 0; t < n_samples; ++t) {
      for (Index_type j = 0; j < n_elements; ++j) {
         const double v = basis_values[j + t * n_elements];
         for (Index_type i = 0; i < n_components; ++i) {
            c[i + j * n_components] += G(i, t) * v;
         }
      }
   }

   for (Index_type i = 0; i < n_variables; ++i) {
      solver->setObjectiveCoefficient(i, c[i]);
   }
}

template <class DistanceMatrix>
ClpSimplex_affiliations_solver::Status
ClpSimplex_affiliations_solver::update_affiliations(
   const DistanceMatrix& G)
{
   if (G.rows() != n_components) {
      throw std::runtime_error(
         "number of rows does not match number of components");
   }

   update_objective(G);

   // if only one local model, solution is trivial
   if (n_components == 1) {
      return Status::SUCCESS;
   } else {
      solver->setMaximumIterations(max_iterations);
      solver->setLogLevel(verbosity);

      solver->initialSolve();

      if (solver->isProvenOptimal()) {
         return Status::SUCCESS;
      } else {
         return Status::FAIL;
      }
   }
}

template <class Matrix>
void ClpSimplex_affiliations_solver::get_objective_coefficients(
   Matrix& M) const
{
   const std::vector<double> coeffs(get_objective_coefficients());

   const int n_rows = M.rows();
   const int n_cols = M.cols();
   const std::size_t n_entries = n_rows * n_cols;
   if (n_entries != coeffs.size()) {
      throw std::runtime_error(
         "number of matrix elements does not match number of coefficients");
   }

   for (int j = 0; j < n_cols; ++j) {
      for (int i = 0; i < n_rows; ++i) {
         M(i, j) = coeffs[i + j * n_components];
      }
   }
}

template <class Matrix>
void ClpSimplex_affiliations_solver::get_affiliations(
   Matrix& Gamma) const
{
   const int n_rows = Gamma.rows();
   const int n_cols = Gamma.cols();

   if (n_rows != n_components) {
      throw std::runtime_error(
         "number of rows does not match number of components");
   }

   if (n_cols != n_samples) {
      throw std::runtime_error(
         "number of columns does not match number of samples");
   }

   if (n_components == 1) {
      for (Index_type t = 0; t < n_samples; ++t) {
         for (Index_type i = 0; i < n_components; ++i) {
            Gamma(i, t) = 1;
         }
      }
   } else {
      const double* sol = solver->getColSolution();
      for (Index_type t = 0; t < n_samples; ++t) {
         for (Index_type i = 0; i < n_components; ++i) {
            Gamma(i, t) = 0;
            for (Index_type j = 0; j < n_elements; ++j) {
               Gamma(i, t) += sol[i + j * n_components] *
                  basis_values[j + t * n_elements];
            }
         }
      }
   }
}

} // namespace pypbn

#endif
