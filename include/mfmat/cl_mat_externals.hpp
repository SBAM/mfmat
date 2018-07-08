#ifndef MFMAT_CL_MAT_EXTERNALS_HPP_
# define MFMAT_CL_MAT_EXTERNALS_HPP_

# include "cl_mat.hpp"

namespace mfmat
{

  /// @brief adds scalar to each matrix's cell
  template <typename T>
  cl_mat<T> operator+(cl_mat<T> lhs, T rhs);

  /// @brief adds scalar to each matrix's cell, reversed
  template <typename T>
  cl_mat<T> operator+(T lhs, cl_mat<T> rhs);

  /// @brief substracts scalar from each matrix's cell
  template <typename T>
  cl_mat<T> operator-(cl_mat<T> lhs, T rhs);

  /// @brief multiplies each matrix's cell by scalar
  template <typename T>
  cl_mat<T> operator*(cl_mat<T> lhs, T rhs);

  /// @brief multiplies each matrix's cell by scalar, reversed
  template <typename T>
  cl_mat<T> operator*(T lhs, cl_mat<T> rhs);

  /// @brief divides each matrix's cell by scalar
  template <typename T>
  cl_mat<T> operator/(cl_mat<T> lhs, T rhs);

  /**
   * @brief sums two matrices
   * @note if rhs is a row/column, it will be added to each result's row/column
   */
  template <typename T>
  cl_mat<T> operator+(cl_mat<T> lhs, cl_mat<T> rhs);

  /**
   * @brief substracts a matrix from another
   * @note if rhs is a row/column, it will be substracted from each
   *       result's row/column
   */
  template <typename T>
  cl_mat<T> operator-(cl_mat<T> lhs, cl_mat<T> rhs);

  /// @brief transposes given matrix
  template <typename T>
  cl_mat<T> transpose(const cl_mat<T>& arg);

  /// @brief multiplies matrices
  template <typename T>
  cl_mat<T> operator*(const cl_mat<T>& lhs, const cl_mat<T>& rhs);

} // !namespace mfmat

#endif // !MFMAT_CL_MAT_EXTERNALS_HPP_
