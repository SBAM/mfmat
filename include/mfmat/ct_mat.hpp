#ifndef MFMAT_CT_MAT_HPP_
# define MFMAT_CT_MAT_HPP_

# include <array>
# include <optional>

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
    /// @typedef opt shorthand to optional ct_mat
    template<typename T2, std::size_t R2, std::size_t C2>
    using opt = std::optional<ct_mat<T2, R2, C2>>;
    /// @typedef square_rr_t defines square RxR ct_mat
    using square_rr_t = ct_mat<T, R, R>;
    /// @typedef square_cc_t defines square CxC ct_mat
    using square_cc_t = ct_mat<T, C, C>;

  public:
    /// @brief Fills matrix with zeroes
    ct_mat() noexcept;
    /// @brief Constructs an identity matrix
    ct_mat(identity_tag) noexcept;
    /// @brief Constructor based on full matrix initialization list
    template <typename T2, std::size_t R2, std::size_t C2>
    ct_mat(const T2(&arg)[R2][C2]) noexcept;
    /**
     * @brief Initializes this matrix by picking only elements pointed by IDXs
     *        from rhs. Those elements are copied sequentially along rows of
     *        this matrix.
     */
    template <std::size_t R2, std::size_t C2, std::size_t... IDXs>
    ct_mat(const ct_mat<T, R2, C2>& rhs,
           std::index_sequence<IDXs...> seq) noexcept;

    ct_mat(const ct_mat&) noexcept = default;
    ct_mat(ct_mat&&) noexcept = default;
    ct_mat& operator=(const ct_mat&) noexcept = default;
    ct_mat& operator=(ct_mat&&) noexcept = default;
    ~ct_mat() noexcept = default;

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
    constexpr T get() const;
    /**
     * @brief constant compile time getter
     * @note equivalent to get<R_IDX, C_IDX> when op_way is by row
     * @tparam OW operation way, specifies first index direction
     * @tparam IDX1 first index, defined by previous operation way
     * @tparam IDX2 second index for specified row or column
     */
    template <op_way OW, std::size_t IDX1, std::size_t IDX2>
    constexpr T get() const;
    /**
     * @brief constant compile time getter using global index over matrix
     * @tparam OW operation way, scan matrix by rows or columns
     * @tparam IDX global index
     */
    template <op_way OW, std::size_t IDX>
    constexpr T scan() const;

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
    constexpr T& get();
    /**
     * @brief modify compile time getter
     * @note equivalent to get<R_IDX, C_IDX> when op_way is by row
     * @tparam OW operation way, specifies first index direction
     * @tparam IDX1 first index, defined be previous operation way
     * @tparam IDX2 second index for specified row or column
     */
    template <op_way OW, std::size_t IDX1, std::size_t IDX2>
    constexpr T& get();
    /**
     * @brief modify compile time getter using global index over matrix
     * @tparam OW operation way, scan matrix by rows or columns
     * @tparam IDX global index
     */
    template <op_way OW, std::size_t IDX>
    constexpr T& scan();

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
    /// @return true if matrices are different
    bool operator!=(const ct_mat& rhs) const noexcept;

    /// @return true if matrix is diagonal
    bool is_diagonal() const noexcept;
    /// @return true if matrix is diagonal based on a provided tolerance
    template <typename = std::enable_if_t<std::is_floating_point_v<T>>>
    bool is_diagonal(T eps_multiplier) const noexcept;

    /// @return true if matrix is symmetric
    bool is_symmetric() const noexcept;

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
     * @return this matrix ortho-normalized in-place, using Gram-Schmidt
     * @warning high complexity
     */
    ct_mat& orthonormalize() noexcept;

    /**
     * @return this square matrix, in-place transposed
     * @note assumes this matrix is square
     */
    ct_mat& transpose() noexcept;

    /// @return matrix' trace
    T trace() const noexcept;

    /**
     * @return matrix determinant
     * @warning recursive method, very high complexity
     */
    T rec_det() const noexcept;

    /**
     * @brief Centers matrix according to columns' means, modifies this matrix.
     * @param pc_mean pre-computed mean, if available, speeds up computation
     */
    ct_mat& mean_center(const opt<T, 1, C>& pc_mean = std::nullopt) noexcept;

    /**
     * @brief Centers matrix according to columns' standard deviations, modifies
     *        this matrix.
     * @param pc_mean pre-computed mean, if available, speeds up computation
     * @param pc_stdd pre-computed standard deviation, if available, speeds up
     *                computation
     */
    ct_mat& stddev_center(const opt<T, 1, C>& pc_mean = std::nullopt,
                          const opt<T, 1, C>& pc_stdd = std::nullopt) noexcept;

  private:
    storage_t storage_; ///< internal storage
  };

  /// @typedef ct_mat_opt shorthand to optional ct_mat
  template<typename T, std::size_t R, std::size_t C>
  using ct_mat_opt = std::optional<ct_mat<T, R, C>>;


  /// @typedef is_ct_mat type traits helper for ct_mat, false_type
  template <typename T>
  struct is_ct_mat : std::false_type {};

  /// @typedef is_ct_mat type traits helper for ct_mat, true_type
  template <typename T, std::size_t R, std::size_t C>
  struct is_ct_mat<ct_mat<T, R, C>> : std::true_type {};

} // !namespace mfmat

# include "ct_mat.ipp"

#endif // !MFMAT_CT_MAT_HPP_
