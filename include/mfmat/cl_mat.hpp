#ifndef MFMAT_CL_MAT_HPP_
# define MFMAT_CL_MAT_HPP_

# include "common.hpp"
# include "cl_kernels_store.hpp"

namespace mfmat
{

  /**
   * @brief Dense matrix that relies on OpenCL kernels for expensive operations.
   * @tparam T matrix cells' type
   */
  template <typename T>
  class cl_mat
  {
  public:
    /// @typedef cell_t shorthand to cell's type
    using cell_t = T;
    /// @typedef storage_t shorthand to storage type
    using storage_t = std::vector<T>;
    /// @typedef iterator shorthand to storage iterator
    using iterator = typename storage_t::iterator;
    /// @typedef const_iterator shorthand to storage const_iterator
    using const_iterator = typename storage_t::const_iterator;

  public:
    /**
     * @brief Fills rectangular matrix with zeroes
     * @param rows matrix rows count
     * @param cols matrix cols count
     */
    cl_mat(std::size_t rows, std::size_t cols);

    /**
     * @brief Fills square matrix with zeroes
     * @param rows_cols matrix rows & cols
     */
    cl_mat(std::size_t rows_cols);

    /**
     * @brief Constructs an identity matrix
     * @param rows_cols matrix rows & cols
     */
    cl_mat(std::size_t rows_cols, identity_tag);

    /// @brief Move constructor that zero out rhs dimensions
    cl_mat(cl_mat&& rhs);
    /// @brief Assignement operator that zero out rhs dimensions
    cl_mat& operator=(cl_mat&& rhs);

    cl_mat(const cl_mat&) = default;
    cl_mat& operator=(const cl_mat&) = default;
    ~cl_mat() = default;

    /// @return matrix's row count
    inline std::size_t get_row_count() const;
    /// @return matrix's column count
    inline std::size_t get_col_count() const;

    /**
     * @note Following iterator helpers simply expose underlying vector's
     *       iterators. This allows using for range based loop on cl_mat.
     */
    inline iterator begin() noexcept;
    inline const_iterator begin() const noexcept;
    inline const_iterator cbegin() const noexcept;
    inline iterator end() noexcept;
    inline const_iterator end() const noexcept;
    inline const_iterator cend() const noexcept;

    /**
     * @brief constant getter using indices pair
     * @param idx idx.first=row_idx, idx.second=col_idx
     */
    inline T operator[](indices idx) const;
    /**
     * @brief constant getter
     * @param row_idx row index
     * @param col_idx column index
     */
    inline T get(std::size_t row_idx, std::size_t col_idx) const;

    /**
     * @brief modify getter using indices pair
     * @param idx idx.first=row_idx, idx.second=col_idx
     */
    inline T& operator[](indices idx);
    /**
     * @brief modify getter
     * @tparam row_idx row index
     * @tparam col_idx column index
     */
    inline T& get(std::size_t row_idx, std::size_t col_idx);

    /// @brief adds scalar and stores result
    cl_mat& operator+=(T val);
    /// @brief substracts scalar and stores result
    cl_mat& operator-=(T val);
    /// @brief multiplies by scalar and stores result
    cl_mat& operator*=(T val);
    /// @brief divides by scalar and stores result
    cl_mat& operator/=(T val);

    /**
     * @brief adds matrix and stores result
     * @note if rhs is a row/column, it will be added to each row/column of lhs
     */
    cl_mat& operator+=(const cl_mat& rhs);
    /**
     * @brief substracts matrix and stores result
     * @note if rhs is a row/column, it will be substracted from each
     *       row/column of lhs
     */
    cl_mat& operator-=(const cl_mat& rhs);


  private:
    std::size_t row_count_; ///< keeps track of current row count
    std::size_t col_count_; ///< keeps track of current column count
    storage_t storage_; ///< internal storage
  };

  /// @typedef cl_mat_f shorthand to float specialized cl_mat
  using cl_mat_f = cl_mat<float>;
  /// @typedef cl_mat_d shorthand to double specialized cl_mat
  using cl_mat_d = cl_mat<double>;

  /// @typedef cl_mat_opt shorthand to optional cl_mat
  template <typename T>
  using cl_mat_opt = std::optional<cl_mat<T>>;

} // !namespace mfmat

# include "cl_mat.ipp"

#endif // !MFMAT_CL_MAT_HPP_