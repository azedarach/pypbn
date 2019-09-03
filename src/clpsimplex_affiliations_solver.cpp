#include "clpsimplex_affiliations_solver.hpp"

#include <CoinBuild.hpp>

#include <cfloat>

namespace pypbn {

const double ClpSimplex_affiliations_solver::Infinity = DBL_MAX;
const double ClpSimplex_affiliations_solver::Minimize = 1;
const double ClpSimplex_affiliations_solver::Maximize = -1;

ClpSimplex_affiliations_solver::Index_type
ClpSimplex_affiliations_solver::get_n_primary_variables() const
{
   return n_components * n_elements;
}

ClpSimplex_affiliations_solver::Index_type
ClpSimplex_affiliations_solver::get_n_auxiliary_variables() const
{
   if (max_tv_norm < 0) {
      return 0;
   } else {
      return n_components * n_elements;
   }
}

std::vector<double> ClpSimplex_affiliations_solver::get_objective_coefficients() const
{
   const int n_primary_variables = get_n_primary_variables();
   const double* coeffs = solver->getObjCoefficients();
   return std::vector<double>(coeffs, coeffs + n_primary_variables);
}

void ClpSimplex_affiliations_solver::initialize_equality_constraints()
{
   for (Index_type t = 0; t < n_samples; ++t) {
      std::vector<int> indices;
      std::vector<double> coefficients;

      for (Index_type j = 0; j < n_elements; ++j) {
         const double v = basis_values[j + t * n_elements];
         if (v == 0) {
            continue;
         }

         for (Index_type i = 0; i < n_components; ++i) {
            indices.push_back(i + j * n_components);
            coefficients.push_back(v);
         }
      }

      if (indices.size() > 0) {
         equality_constraints.emplace_back(
            indices, coefficients, 1, 1);
      }
   }
}

void ClpSimplex_affiliations_solver::initialize_basic_inequality_constraints()
{
   for (Index_type i = 0; i < n_components; ++i) {
      for (Index_type t = 0; t < n_samples; ++t) {
         std::vector<int> indices;
         std::vector<double> coefficients;

         for (Index_type j = 0; j < n_elements; ++j) {
            if (basis_values[j + t * n_elements] != 0) {
               indices.push_back(i + j * n_components);
               coefficients.push_back(basis_values[j + t * n_elements]);
            }
         }

         const auto is_trivial_bound = indices.size() == 1;
         if (is_trivial_bound) {
            const int idx = indices[0];
            if (coefficients[0] < 0) {
               solver->setColumnLower(idx, -ClpSimplex_affiliations_solver::Infinity);
               solver->setColumnUpper(idx, 0);
            } else {
               solver->setColumnLower(idx, 0);
               solver->setColumnUpper(idx, ClpSimplex_affiliations_solver::Infinity);
            }
         } else {
            positivity_constraints.emplace_back(
               indices, coefficients,
               0, ClpSimplex_affiliations_solver::Infinity);
         }
      }
   }
}

void ClpSimplex_affiliations_solver::initialize_auxiliary_inequality_constraints()
{
   const auto n_primary_vars = n_components * n_elements;

   for (Index_type i = 0; i < n_components; ++i) {
      for (Index_type t = 0; t < n_samples - 1; ++t) {
         // indices and coefficients corresponding to 
         // the n_components * (n_samples - 1) constraints
         // -\gamma_i(t+1) + \gamma_i(t) - \eta_i(t) \leq 0
         std::vector<int> lower_indices;
         std::vector<double> lower_coefficients;

         // indices and coefficients corresponding to 
         // the n_components * (n_samples - 1) constraints
         // \gamma_i(t+1) - \gamma_i(t) - \eta_i(t) \leq 0
         std::vector<int> upper_indices;
         std::vector<double> upper_coefficients;

         // indices and coefficients corresponding to the
         // n_components * (n_samples - 1) constraints
         // \eta_i(t) \geq 0
         std::vector<int> aux_indices;
         std::vector<double> aux_coefficients;

         for (Index_type j = 0; j < n_elements; ++j) {
            const double vt = basis_values[j + t * n_elements];
            const double vtp1 = basis_values[j + (t + 1) * n_elements];
            const double dv = vtp1 - vt;

            if (dv != 0) {
               lower_indices.push_back(i + j * n_components);
               lower_coefficients.push_back(-dv);

               upper_indices.push_back(i + j * n_components);
               upper_coefficients.push_back(dv);
            }

            if (vt != 0) {
               lower_indices.push_back(i + j * n_components + n_primary_vars);
               lower_coefficients.push_back(-vt);

               upper_indices.push_back(i + j * n_components + n_primary_vars);
               upper_coefficients.push_back(-vt);

               aux_indices.push_back(i + j * n_components + n_primary_vars);
               aux_coefficients.push_back(vt);
            }
         }

         aux_positivity_constraints.emplace_back(
            lower_indices, lower_coefficients,
            -ClpSimplex_affiliations_solver::Infinity, 0);

         aux_positivity_constraints.emplace_back(
            upper_indices, upper_coefficients,
            -ClpSimplex_affiliations_solver::Infinity, 0);

         if (aux_indices.size() == 1) {
            // for the case that only one basis element v_j is non-zero
            // at a given time, the constraint \eta_i(t) \geq 0
            // reduces to a bound on the corresponding FE coefficient,
            // B_{ij} v_j(t) \geq 0, i.e., B_{ij} \leq 0 if v_j(t) < 0
            // and B_{ij} \geq 0 otherwise
            const int idx = aux_indices[0];
            if (aux_coefficients[0] < 0) {
               solver->setColumnLower(idx, -ClpSimplex_affiliations_solver::Infinity);
               solver->setColumnUpper(idx, 0);
            } else {
               solver->setColumnLower(idx, 0);
               solver->setColumnUpper(idx, ClpSimplex_affiliations_solver::Infinity);
            }
         } else {
            if (aux_indices.size() > 1) {
               aux_positivity_constraints.emplace_back(
                  aux_indices, aux_coefficients,
                  0, ClpSimplex_affiliations_solver::Infinity);
            }
         }
      }
   }

   // indices and coefficients corresponding to the n_components
   // constraints \sum_{t = 1}^{n_samples - 1} \eta_i(t) \leq max_tv_norm
   std::vector<std::vector<int> > indices(n_components);
   std::vector<std::vector<double> > coefficients(n_components);
   for (Index_type t = 0; t < n_samples - 1; ++t) {
      for (Index_type j = 0; j < n_elements; ++j) {
         const double v = basis_values[j + t * n_elements];
         if (v == 0) {
            continue;
         }
         for (Index_type i = 0; i < n_components; ++i) {
            indices[i].push_back(i + j * n_components + n_primary_vars);
            coefficients[i].push_back(v);
         }
      }
   }

   for (Index_type i = 0; i < n_components; ++i) {
      aux_norm_constraints.emplace_back(
         indices[i], coefficients[i],
         -ClpSimplex_affiliations_solver::Infinity, max_tv_norm);
   }
}

void ClpSimplex_affiliations_solver::add_constraints()
{
   CoinBuild build;

   for (auto& c : equality_constraints) {
      build.addRow(
         c.size(), c.get_indices(), c.get_coefficients(),
         c.lower_bound, c.upper_bound);
   }

   for (auto& c : positivity_constraints) {
      build.addRow(
         c.size(), c.get_indices(), c.get_coefficients(),
         c.lower_bound, c.upper_bound);
   }

   if (!aux_positivity_constraints.empty()) {
      for (auto& c : aux_positivity_constraints) {
         build.addRow(
            c.size(), c.get_indices(), c.get_coefficients(),
            c.lower_bound, c.upper_bound);
      }
   }

   if (!aux_norm_constraints.empty()) {
      for (auto& c : aux_norm_constraints) {
         build.addRow(
            c.size(), c.get_indices(), c.get_coefficients(),
            c.lower_bound, c.upper_bound);
      }
   }

   solver->addRows(build);
}

void ClpSimplex_affiliations_solver::initialize_constraints_and_bounds()
{
   initialize_equality_constraints();
   initialize_basic_inequality_constraints();
   if (max_tv_norm >= 0) {
      initialize_auxiliary_inequality_constraints();
   }
   add_constraints();
}

} // namespace pypbn
