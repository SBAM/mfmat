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

    cl_mat(const cl_mat&) = default;
    cl_mat(cl_mat&&) = default;
    cl_mat& operator=(const cl_mat&) = default;
    cl_mat& operator=(cl_mat&&) = default;
    ~cl_mat() = default;

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

#endif // !MFMAT_CL_MAT_HPP_
