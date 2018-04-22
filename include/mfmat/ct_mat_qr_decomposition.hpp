#ifndef MFMAT_CT_MAT_QR_DECOMPOSITION_
# define MFMAT_CT_MAT_QR_DECOMPOSITION_

# include "ct_mat_externals.hpp"

namespace mfmat
{

  /**
   * @brief This helper class QR decomposes given matrix and provides
   *        accessors to decomposition parts.
   *         - Q is orthonormal (internally uses Gram-Schmidt), its size
   *           matches original input
   *         - R is upper triangular
   */
  template <typename T, std::size_t R, std::size_t C>
  class qr_decomposition
  {
  public:
    /// @typedef m_t input matrix type
    using m_t = ct_mat<T, R, C>;
    /// @typedef q_t Q matrix type
    using q_t = m_t;
    /// @typedef r_t R matrix type
    using r_t = ct_mat<T, C, C>;

  public:
    /// @brief Default initializes Q and R to zero
    qr_decomposition() noexcept = default;
    /// @brief Invokes operator() on input and stores result
    qr_decomposition(const m_t& input) noexcept;
    ~qr_decomposition() noexcept = default;

    qr_decomposition(const qr_decomposition&) = delete;
    qr_decomposition(qr_decomposition&&) = delete;
    qr_decomposition& operator=(const qr_decomposition&) = delete;

    /**
     * @brief QR decomposes input.
     * @note Q is input orthonormalized
     *       R is transpose(Q) * input
     */
    qr_decomposition& operator()(const m_t& input) noexcept;

    /// @return stored Q
    constexpr q_t& get_q();
    /// @return const stored Q
    constexpr const q_t& get_q() const;

    /// @return stored R
    constexpr r_t& get_r();
    /// @return const stored R
    constexpr const r_t& get_r() const;

  private:
    q_t q_; ///< Q part of QR decomposition
    r_t r_; ///< R part of QR decomposition
  };

} // !namespace mfmat

# include "ct_mat_qr_decomposition.ipp"

#endif // !MFMAT_CT_MAT_QR_DECOMPOSITION_
