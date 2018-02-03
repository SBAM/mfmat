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
    constexpr dense_matrix(matrix_init_list<T> mil) noexcept;

    /// @brief Constructor based on single column initialization list
    dense_matrix(column_init_list<T> cil) noexcept;

    dense_matrix(const dense_matrix&) = default;
    dense_matrix(dense_matrix&&) = default;
    dense_matrix& operator=(const dense_matrix&) = default;
    ~dense_matrix() = default;

    /// @brief constant runtime getter using indices
    constexpr const T& operator[](indices idx) const noexcept;

    /// @brief constant compile time getter
    template <std::size_t I, std::size_t J>
    constexpr const T& get() const noexcept;

    /// @brief runtime getter using indices
    constexpr T& operator[](indices idx) noexcept;

    /// @brief compile time getter
    template <std::size_t I, std::size_t J>
    constexpr T& get() noexcept;

  private:
    storage_t storage_; ///< internal storage
  };

} // !namespace mfmat

# include "dense_matrix.ipp"

#endif // !MFMAT_DENSE_MATRIX_HPP_
