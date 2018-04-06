#ifndef MFMAT_CT_MAT_EXTERNALS_HPP_
# define MFMAT_CT_MAT_EXTERNALS_HPP_

# include "ct_mat.hpp"

namespace mfmat
{

  /// @brief adds scalar to each matrix's cell
  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C> operator+(ct_mat<T, R, C> lhs, T rhs) noexcept;

  /// @brief adds scalar to each matrix's cell, reversed
  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C> operator+(T lhs, ct_mat<T, R, C> rhs) noexcept;

  /// @brief substracts scalar from each matrix's cell
  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C> operator-(ct_mat<T, R, C> lhs, T rhs) noexcept;

  /// @brief multiplies each matrix's cell by scalar
  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C> operator*(ct_mat<T, R, C> lhs, T rhs) noexcept;

  /// @brief multiplies each matrix's cell by scalar, reversed
  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C> operator*(T lhs, ct_mat<T, R, C> rhs) noexcept;

  /// @brief divides each matrix's cell by scalar
  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C> operator/(ct_mat<T, R, C> lhs, T rhs) noexcept;

  /// @brief sums two matrices
  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C>
  operator+(ct_mat<T, R, C> lhs, const ct_mat<T, R, C>& rhs) noexcept;

  /// @brief substracts a matrix from another
  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C>
  operator-(ct_mat<T, R, C> lhs, const ct_mat<T, R, C>& rhs) noexcept;

  /**
   * @brief copies a vector (row or column) from a source matrix to a vector
   *        (row or column) in given destination matrix
   * @tparam OW_SRC defines if copy source is a row or column
   * @tparam IDX_SRC row/col index of vector to copy
   * @tparam OW_DST defines if copy destination is a row or column
   * @tparam IDX_DST row/col index of copy destination
   * @tparam M_SRC source matrix type
   * @tparam M_DST destination matrix type
   */
  template <op_way OW_SRC, std::size_t IDX_SRC,
            op_way OW_DST, std::size_t IDX_DST,
            typename M_SRC, typename M_DST>
  void copy_vector(const M_SRC& src, M_DST& dst) noexcept;

  /// @brief transposes given matrix
  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, C, R> transpose(const ct_mat<T, R, C>& arg) noexcept;

  /**
   * @tparam OW1 defines if dot product's LHS is a row or column
   * @tparam IDX1 row/col index of dot product's LHS
   * @tparam OW2 defines if dot product's RHS is a row or column
   * @tparam IDX2 row/col index of dot product's RHS
   * @tparam M1 first matrix type
   * @tparam M2 second matrix type
   * @return dot product of specified row/col
   */
  template <op_way OW1, std::size_t IDX1,
            op_way OW2, std::size_t IDX2,
            typename M1, typename M2>
  auto dot(const M1& mat1, const M2& mat2) noexcept;

  /// @brief multiplies matrices
  template <typename M1, typename M2>
  auto operator*(const M1& lhs, const M2& rhs) noexcept;

  /**
   * @warning high complexity
   * @return this matrix ortho-normalized using Gram-Schmidt
   */
  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C> orthonormalize(const ct_mat<T, R, C>& arg) noexcept;

  /**
   * @brief Builds a vector that stores the mean value of columns
   * @return a row of size C
   */
  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, 1, C> mean(const ct_mat<T, R, C>& arg) noexcept;


  /**
   * @brief Builds a vector that stores the standard deviation of each column
   * @param pc_mean pre-computed mean, if available, speeds up computation
   * @return a row of size C
   */
  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, 1, C> std_dev(const ct_mat<T, R, C>& arg,
                          const ct_mat_opt<T, 1, C>& pc_mean = std::nullopt);

  /**
   * @brief Builds the deviation matrix according to the mean computed against
   *        rows or columns.
   * @tparam OW defines if mean is computed along rows or columns
   * @return deviation matrix with dimensions matching original arg
   */
  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C> deviation(const ct_mat<T, R, C>& arg) noexcept;

  /**
   * @brief Computes covariance of a data-set where observations are stored as
   *        rows.
   * @return matrix normalized by the number of observations
   */
  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, C, C> covariance(const ct_mat<T, R, C>& arg) noexcept;

} // !namespace mfmat

# include "ct_mat_externals.ipp"

#endif // !MFMAT_CT_MAT_EXTERNALS_HPP_
