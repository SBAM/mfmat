#ifndef MFMAT_CT_MAT_QR_EIGEN_HPP_
# define MFMAT_CT_MAT_QR_EIGEN_HPP_

# include "ct_mat_qr_decomposition.hpp"

namespace mfmat
{

  /**
   * @brief Extracts eigen values and eigen vectors from given symmetric matrix
   *        using QR decompositions.
   *        Square input has dimension D.
   */
  template <typename T, std::size_t D>
  class qr_eigen
  {
  public:
    /// @typedef m_t working matrix type
    using m_t = ct_mat<T, D, D>;

  public:
    /// @brief Default initializes eigen values/vectors to zero
    qr_eigen() noexcept = default;
    /// @brief Invokes operator() with iteration limitation
    qr_eigen(const m_t& input, std::size_t max_iter = 32) noexcept;
    ~qr_eigen() noexcept = default;

    qr_eigen(const qr_eigen&) = delete;
    qr_eigen(qr_eigen&&) = delete;
    qr_eigen& operator=(const qr_eigen&) = delete;

    /**
     * @brief Extracts eigen values/vectors from input. This is an iterative
     *        procedure that can either stop when:
     *         - values converges to a diagonal matrix
     *         - values no longer changes between iterations
     *         - max_iter is reached.
     */
    qr_eigen& operator()(const m_t& input, std::size_t max_iter = 32) noexcept;

    /// @return stored eigen values
    constexpr m_t& get_values();
    /// @return const stored eigen values
    constexpr const m_t& get_values() const;

    /// @return stored eigen vectors
    constexpr m_t& get_vectors();
    /// @return const stored eigen vectors
    constexpr const m_t& get_vectors() const;

  private:
    m_t values_; ///< eigen values
    m_t prev_values_; ///< previous iteration eigen values
    m_t vectors_; ///< eigen vectors
    qr_decomposition<T, D, D> decomp_; ///< underlying QR decomposition
  };

} // !namespace mfmat

# include "ct_mat_qr_eigen.ipp"

#endif // !
