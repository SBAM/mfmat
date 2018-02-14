#ifndef MFMAT_DENSE_MATRIX_HPP_
# define MFMAT_DENSE_MATRIX_HPP_

# include <array>

# include "common.hpp"

namespace mfmat
{

  /**
   * @brief Basic dense matrix.
   * @tparam T matrix cells' type
   * @tparam R rows count
   * @tparam columns count
   */
  template <typename T, std::size_t R, std::size_t C>
  class dense_matrix
  {
  public:
    template<typename, std::size_t, std::size_t>
    friend class dense_matrix;
    static constexpr std::size_t row_count = R; ///< row count accessor
    static constexpr std::size_t col_count = C; ///< column count accessor
    /// @typedef cell_t shorthand to cell's type
    using cell_t = T;
    /// @typedef storage_t Shorthand to internal storage type
    using storage_t = std::array<std::array<T, C>, R>;

  public:
    /// @brief Fills matrix with zeroes
    dense_matrix() noexcept;
    /// @brief Constructs an identity matrix
    dense_matrix(identity_tag) noexcept;
    /// @brief Constructor based on full matrix initialization list
    template <typename T2, std::size_t R2, std::size_t C2>
    dense_matrix(const T2(&cil)[R2][C2]) noexcept;

    dense_matrix(const dense_matrix&) = default;
    dense_matrix(dense_matrix&&) = default;
    dense_matrix& operator=(const dense_matrix&) = default;
    ~dense_matrix() = default;

    /// @brief constant runtime getter using indices
    constexpr T operator[](indices idx) const noexcept;
    /// @brief constant compile time getter (row & column index)
    template <std::size_t I, std::size_t J>
    constexpr T get() const noexcept;
    /// @brief constant compile time getter (traverse by rows)
    template <std::size_t I>
    constexpr T scan_r() const noexcept;
    /// @brief constant compile time getter (traverse by columns)
    template <std::size_t I>
    constexpr T scan_c() const noexcept;
    /// @brief runtime getter using indices
    constexpr T& operator[](indices idx) noexcept;
    /// @brief compile time getter (row & column index)
    template <std::size_t I, std::size_t J>
    constexpr T& get() noexcept;
    /// @brief compile time getter (traverse by rows)
    template <std::size_t I>
    constexpr T& scan_r() noexcept;
    /// @brief compile time getter (traverse by columns)
    template <std::size_t I>
    constexpr T& scan_c() noexcept;

    /// @brief adds scalar and stores result
    dense_matrix& operator+=(T val) noexcept;
    /// @brief substracts scalar and stores result
    dense_matrix& operator-=(T val) noexcept;
    /// @brief multiplies by scalar and stores result
    dense_matrix& operator*=(T val) noexcept;
    /// @brief divides by scalar and stores result
    dense_matrix& operator/=(T val) noexcept;

    /// @brief adds matrix and stores result
    dense_matrix& operator+=(const dense_matrix& rhs) noexcept;
    /// @brief substracts matrix and stores result
    dense_matrix& operator-=(const dense_matrix& rhs) noexcept;

    /// @brief multiplies matrices
    template <typename T2, std::size_t R2, std::size_t C2>
    auto operator*(const dense_matrix<T2, R2, C2>& rhs) const noexcept;

    /// @return true if both matrices are equal
    bool operator==(const dense_matrix& rhs) const noexcept;
    /// @return true if both matrices are different
    bool operator!=(const dense_matrix& rhs) const noexcept;

    /// @return transposed matrix
    auto transpose() const noexcept;
    /// @return matrix' trace
    constexpr T trace() const noexcept;
    /**
     * @return matrix determinant
     * @warning recursive method, very high complexity
     */
    constexpr T rec_det() const noexcept;

  private:
    storage_t storage_; ///< internal storage
  };


  /// @brief external addition operator
  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>
  operator+(const dense_matrix<T, R, C>& lhs, T rhs) noexcept;

  /// @brief external addition operator, commutative
  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>
  operator+(T lhs, const dense_matrix<T, R, C>& rhs) noexcept;

  /// @brief external substraction operator
  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>
  operator-(const dense_matrix<T, R, C>& lhs, T rhs) noexcept;

    /// @brief external multiply operator
  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>
  operator*(const dense_matrix<T, R, C>& lhs, T rhs) noexcept;

  /// @brief external multiply operator, commutative
  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>
  operator+(T lhs, const dense_matrix<T, R, C>& rhs) noexcept;

  /// @brief external divide operator
  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>
  operator/(const dense_matrix<T, R, C>& lhs, T rhs) noexcept;

    /// @brief external addition operator
  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>
  operator+(const dense_matrix<T, R, C>& lhs,
            const dense_matrix<T, R, C>& rhs) noexcept;

  /// @brief external substraction operator
  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>
  operator-(const dense_matrix<T, R, C>& lhs,
            const dense_matrix<T, R, C>& rhs) noexcept;

} // !namespace mfmat

# include "dense_matrix.ipp"
# include "dense_matrix_externals.ipp"

#endif // !MFMAT_DENSE_MATRIX_HPP_
