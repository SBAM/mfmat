namespace mfmat
{

  template <typename T>
  qr_eigen<T>::qr_eigen(const m_t& input, std::size_t max_iter)
  {
    this->operator()(input, max_iter);
  }


  template <typename T>
  qr_eigen<T>&
  qr_eigen<T>::operator()(const m_t& input, std::size_t max_iter)
  {
    if (!input.is_symmetric())
      throw std::runtime_error("qr_eigen::operator() matrix must be symmetric");
    values_ = input;
    if constexpr (is_ct_mat<T>::value)
      vectors_ = m_t(identity_tag{});
    else
      vectors_ = m_t(input.get_row_count(), identity_tag{});
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


  template <typename T>
  constexpr typename qr_eigen<T>::m_t&
  qr_eigen<T>::get_values()
  {
    return values_;
  }


  template <typename T>
  constexpr const typename qr_eigen<T>::m_t&
  qr_eigen<T>::get_values() const
  {
    return values_;
  }


  template <typename T>
  constexpr typename qr_eigen<T>::m_t&
  qr_eigen<T>::get_vectors()
  {
    return vectors_;
  }


  template <typename T>
  constexpr const typename qr_eigen<T>::m_t&
  qr_eigen<T>::get_vectors() const
  {
    return vectors_;
  }

} // !namespace mfmat
