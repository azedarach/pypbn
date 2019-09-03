#include "linear_constraint.hpp"

#include <stdexcept>

namespace pypbn {

Linear_constraint::Linear_constraint(
   const Index_vector& indices_,
   const Coefficients_vector& coefficients_,
   double lower_bound_, double upper_bound_)
   : lower_bound(lower_bound_)
   , upper_bound(upper_bound_)
{
   if (indices_.size() != coefficients_.size()) {
      throw std::runtime_error(
         "number of indices does not match number of coefficients");
   }

   n_coefficients = indices_.size();
   indices = std::unique_ptr<Index_vector>(new Index_vector(indices_));
   coefficients = std::unique_ptr<Coefficients_vector>(new Coefficients_vector(coefficients_));
}

} // namespace pypbn
