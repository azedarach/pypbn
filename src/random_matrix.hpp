#ifndef CMIPFEMBVBIN_RANDOM_MATRIX_HPP_INCLUDED
#define CMIPFEMBVBIN_RANDOM_MATRIX_HPP_INCLUDED

/**
 * @file random_matrix.hpp
 * @brief provides utilities for generating random and stochastic matrices
 */

#include <random>
#include <vector>

namespace cmip_fembv {

/**
 * @brief normalize matrix to be left stochastic
 * @tparam Matrix matrix type
 * @param A the matrix to be normalized
 */
template <class Matrix>
void make_left_stochastic(Matrix& A)
{
   const int n_rows = A.rows();
   const int n_cols = A.cols();

   std::vector<double> col_sums(n_cols, 0);
   for (int j = 0; j < n_cols; ++j) {
      for (int i = 0; i < n_rows; ++i) {
         col_sums[j] += A(i, j);
      }
   }

   for (int j = 0; j < n_cols; ++j) {
      for (int i = 0; i < n_rows; ++i) {
         A(i, j) /= col_sums[j];
      }
   }
}

/**
 * @brief normalize matrix to be right stochastic
 * @tparam Matrix matrix type
 * @param A the matrix to be normalized
 */
template <class Matrix>
void make_right_stochastic(Matrix& A)
{
   const int n_rows = A.rows();
   const int n_cols = A.cols();

   std::vector<double> row_sums(n_rows, 0);
   for (int j = 0; j < n_cols; ++j) {
      for (int i = 0; i < n_rows; ++i) {
         row_sums[i] += A(i, j);
      }
   }

   for (int j = 0; j < n_cols; ++j) {
      for (int i = 0; i < n_rows; ++i) {
         A(i, j) /= row_sums[i];
      }
   }
}

/**
 * @brief fill random left stochastic matrix
 * @tparam Matrix matrix type
 * @tparam Generator pseudo-random number generator type
 * @param A matrix to store result
 * @param generator pseudo-random number generator
 */
template <class Matrix, class Generator>
void random_left_stochastic_matrix(Matrix& A, Generator& generator)
{
   std::uniform_real_distribution<> dist(0., 1.);

   const int n_rows = A.rows();
   const int n_cols = A.cols();

   std::vector<double> col_sums(n_cols, 0);
   for (int j = 0; j < n_cols; ++j) {
      for (int i = 0; i < n_rows; ++i) {
         const auto aij = dist(generator);
         A(i, j) = aij;
         col_sums[j] += aij;
      }
   }

   for (int j = 0; j < n_cols; ++j) {
      for (int i = 0; i < n_rows; ++i) {
         A(i, j) /= col_sums[j];
      }
   }
}

/**
 * @brief fill random right stochastic matrix
 * @tparam Matrix matrix type
 * @tparam Generator pseudo-random number generator type
 * @param A matrix to store result
 * @param generator pseudo-random number generator
 */
template <class Matrix, class Generator>
void random_right_stochastic_matrix(Matrix& A, Generator& generator)
{
   std::uniform_real_distribution<> dist(0., 1.);

   const int n_rows = A.rows();
   const int n_cols = A.cols();

   std::vector<double> row_sums(n_rows, 0);
   for (int j = 0; j < n_cols; ++j) {
      for (int i = 0; i < n_rows; ++i) {
         const auto aij = dist(generator);
         A(i, j) = aij;
         row_sums[i] += aij;
      }
   }

   for (int j = 0; j < n_cols; ++j) {
      for (int i = 0; i < n_rows; ++i) {
         A(i, j) /= row_sums[i];
      }
   }
}

} // namespace cmip_fembv

#endif
