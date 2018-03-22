#ifndef MFMAT_CT_MAT_HPP_
# define MFMAT_CT_MAT_HPP_

# include <array>

# include "common.hpp"
# include "ct_sequence_helpers.hpp"

namespace mfmat
{

  /**
   * @brief Dense matrix with compile time defined dimensions
   * @tparam T matrix cells' type
   * @tparam R rows count
   * @tparam C columns count
   */
  template <typename T, std::size_t R, std::size_t C>
  class ct_mat
  {
  public:
    template<typename, std::size_t, std::size_t>
    friend class ct_mat;
    static constexpr std::size_t row_count = R; ///< row count accessor
    static constexpr std::size_t col_count = C; ///< column count accessor
    /// @typedef cell_t shorthand to cell's type
    using cell_t = T;
    /// @typedef storage_t Shorthand to internal storage type
    using storage_t = std::array<std::array<T, C>, R>;

  public:
    /// @brief Fills matrix with zeroes
    ct_mat() noexcept;
    /// @brief Constructs an identity matrix
    ct_mat(identity_tag) noexcept;
    /// @brief Constructor based on full matrix initialization list
    template <typename T2, std::size_t R2, std::size_t C2>
    ct_mat(const T2(&cil)[R2][C2]) noexcept;

    ct_mat(const ct_mat&) = default;
    ct_mat(ct_mat&&) = default;
    ct_mat& operator=(const ct_mat&) = default;
    ct_mat& operator=(ct_mat&&) = default;
    ~ct_mat() = default;

    /**
     * @brief constant runtime getter using indices pair
     * @param idx idx.first=row_idx, idx.second=col_idx
     */
    T operator[](indices idx) const noexcept;
    /**
     * @brief constant compile time getter
     * @tparam R_IDX row index
     * @tparam C_IDX column index
     */
    template <std::size_t R_IDX, std::size_t C_IDX>
    constexpr T get() const noexcept;
    /**
     * @brief constant compile time getter
     * @note equivalent to get<R_IDX, C_IDX> when op_way is by row
     * @tparam OW operation way, specifies first index direction
     * @tparam IDX1 first index, defined by previous operation way
     * @tparam IDX2 second index for specified row or column
     */
    template <op_way OW, std::size_t IDX1, std::size_t IDX2>
    constexpr T get() const noexcept;
    /**
     * @brief constant compile time getter using global index over matrix
     * @tparam OW operation way, scan matrix by rows or columns
     * @tparam IDX global index
     */
    template <op_way OW, std::size_t IDX>
    constexpr T scan() const noexcept;

    /**
     * @brief modify runtime getter using indices pair
     * @param idx idx.first=row_idx, idx.second=col_idx
     */
    T& operator[](indices idx) noexcept;
    /**
     * @brief modify compile time getter
     * @tparam R_IDX row index
     * @tparam C_IDX column index
     */
    template <std::size_t R_IDX, std::size_t C_IDX>
    constexpr T& get() noexcept;
    /**
     * @brief modify compile time getter
     * @note equivalent to get<R_IDX, C_IDX> when op_way is by row
     * @tparam OW operation way, specifies first index direction
     * @tparam IDX1 first index, defined be previous operation way
     * @tparam IDX2 second index for specified row or column
     */
    template <op_way OW, std::size_t IDX1, std::size_t IDX2>
    constexpr T& get() noexcept;
    /**
     * @brief modify compile time getter using global index over matrix
     * @tparam OW operation way, scan matrix by rows or columns
     * @tparam IDX global index
     */
    template <op_way OW, std::size_t IDX>
    constexpr T& scan() noexcept;

    /// @brief adds scalar and stores result
    ct_mat& operator+=(T val) noexcept;
    /// @brief substracts scalar and stores result
    ct_mat& operator-=(T val) noexcept;
    /// @brief multiplies by scalar and stores result
    ct_mat& operator*=(T val) noexcept;
    /// @brief divides by scalar and stores result
    ct_mat& operator/=(T val) noexcept;

    /// @brief adds matrix and stores result
    ct_mat& operator+=(const ct_mat& rhs) noexcept;
    /// @brief substracts matrix and stores result
    ct_mat& operator-=(const ct_mat& rhs) noexcept;

    /// @return true if both matrices are equal
    bool operator==(const ct_mat& rhs) const noexcept;
    /// @return true if both matrices are different
    bool operator!=(const ct_mat& rhs) const noexcept;

    /**
     * @return norm of given row/column at IDX location
     * @tparam OW operation way, norm of row or column
     * @tparam IDX row or column index
     */
    template <op_way OW, std::size_t IDX>
    T norm() const noexcept;

    /**
     * @return false if vector length was 0, data then remains untouched
     * @tparam OW operation way, normalize row or column
     * @tparam IDX row or column index to normalize
     */
    template <op_way OW, std::size_t IDX>
    bool normalize() noexcept;

    /**
     * @brief transposes in place this matrix
     * @note assumes this matrix is square
     */
    void transpose() noexcept;

    /// @return matrix' trace
    T trace() const noexcept;

    /**
     * @return matrix determinant
     * @warning recursive method, very high complexity
     */
    constexpr T rec_det() const noexcept;

  private:
    storage_t storage_; ///< internal storage
  };

} // !namespace mfmat

# include "ct_mat.ipp"

#endif // !MFMAT_CT_MAT_HPP_
