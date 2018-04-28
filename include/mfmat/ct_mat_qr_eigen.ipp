namespace mfmat
{

  template <typename T, std::size_t D>
  qr_eigen<T, D>::qr_eigen(const m_t& input, std::size_t max_iter) noexcept
  {
    this->operator()(input, max_iter);
  }


  template <typename T, std::size_t D>
  qr_eigen<T, D>&
  qr_eigen<T, D>::operator()(const m_t& input, std::size_t max_iter) noexcept
  {
    // @warning Input must be symmetric
    assert(input.is_symmetric());
    values_ = input;
    vectors_ = m_t(identity_tag{});
    do
    {
      decomp_(values_);
      prev_values_ = values_;
      values_ = decomp_.get_r() * decomp_.get_q();
      vectors_ = vectors_ * decomp_.get_q();
    }
    while (!(values_.is_diagonal() ||
             values_ == prev_values_ ||
             max_iter-- == 0));
    return *this;
  }


  template <typename T, std::size_t D>
  constexpr typename qr_eigen<T, D>::m_t&
  qr_eigen<T, D>::get_values()
  {
    return values_;
  }


  template <typename T, std::size_t D>
  constexpr const typename qr_eigen<T, D>::m_t&
  qr_eigen<T, D>::get_values() const
  {
    return values_;
  }


  template <typename T, std::size_t D>
  constexpr typename qr_eigen<T, D>::m_t&
  qr_eigen<T, D>::get_vectors()
  {
    return vectors_;
  }


  template <typename T, std::size_t D>
  constexpr const typename qr_eigen<T, D>::m_t&
  qr_eigen<T, D>::get_vectors() const
  {
    return vectors_;
  }

} // !namespace mfmat
