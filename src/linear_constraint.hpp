#ifndef PYPBN_LINEAR_CONSTRAINT_HPP_INCLUDED
#define PYPBN_LINEAR_CONSTRAINT_HPP_INCLUDED

#include <memory>
#include <vector>

namespace pypbn {

class Linear_constraint {
public:
   using Index_vector = std::vector<int>;
   using Coefficients_vector = std::vector<double>;

   double lower_bound{};
   double upper_bound{};

   Linear_constraint(const Index_vector&,
                     const Coefficients_vector&,
                     double, double);
   ~Linear_constraint() = default;
   Linear_constraint(const Linear_constraint&) = delete;
   Linear_constraint(Linear_constraint&&) = default;
   Linear_constraint& operator=(const Linear_constraint&) = delete;
   Linear_constraint& operator=(Linear_constraint&&) = default;

   std::size_t size() const { return n_coefficients; }

   int* get_indices() { return indices->data(); }
   double* get_coefficients() { return coefficients->data(); }

private:
   std::size_t n_coefficients{};
   std::unique_ptr<Index_vector> indices{};
   std::unique_ptr<Coefficients_vector> coefficients{};
};

} // namespace pypbn

#endif