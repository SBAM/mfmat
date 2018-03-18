#ifndef MFMAT_CT_MAT_EXTERNALS_HPP_
# define MFMAT_CT_MAT_EXTERNALS_HPP_

# include "ct_mat.hpp"

namespace mfmat
{

  /// @brief adds scalar to each matrix's cell
  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C> operator+(const ct_mat<T, R, C>& lhs, T rhs) noexcept;

  /// @brief adds scalar to each matrix's cell, commutative version
  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C> operator+(T lhs, const ct_mat<T, R, C>& rhs) noexcept;

  /// @brief substracts scalar from each matrix's cell
  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C> operator-(const ct_mat<T, R, C>& lhs, T rhs) noexcept;

  /// @brief multiplies each matrix's cell by scalar
  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C> operator*(const ct_mat<T, R, C>& lhs, T rhs) noexcept;

  /// @brief multiplies each matrix's cell by scalar, commutative version
  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C> operator+(T lhs, const ct_mat<T, R, C>& rhs) noexcept;

  /// @brief divides each matrix's cell by scalar
  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C> operator/(const ct_mat<T, R, C>& lhs, T rhs) noexcept;

    /// @brief sums two matrices
  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C>
  operator+(const ct_mat<T, R, C>& lhs, const ct_mat<T, R, C>& rhs) noexcept;

  /// @brief substracts a matrix from another
  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C>
  operator-(const ct_mat<T, R, C>& lhs, const ct_mat<T, R, C>& rhs) noexcept;

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

} // !namespace mfmat

# include "ct_mat_externals.ipp"

#endif // !MFMAT_CT_MAT_EXTERNALS_HPP_
