namespace mfmat
{

  template <typename T, std::size_t R, std::size_t C>
  qr_decomposition<T, R, C>::qr_decomposition(const m_t& input) noexcept
  {
    this->operator()(input);
  }


  template <typename T, std::size_t R, std::size_t C>
  qr_decomposition<T, R, C>&
  qr_decomposition<T, R, C>::operator()(const m_t& input) noexcept
  {
    q_ = input;
    q_.orthonormalize();
    // avoids creating temporary transposed q_
    // r_ = transpose(q_) * input;
    auto transpose_mul = [&]<auto... Is>(std::index_sequence<Is...>)
      {
        ((r_.template scan<op_way::row, Is>() =
            dot<op_way::col, Is / r_t::col_count,
                op_way::col, Is % r_t::col_count>(q_, input)), ...);
      };
    transpose_mul(std::make_index_sequence<r_t::row_count * r_t::col_count>{});
    return *this;
  }


  template <typename T, std::size_t R, std::size_t C>
  constexpr typename qr_decomposition<T, R, C>::q_t&
  qr_decomposition<T, R, C>::get_q()
  {
    return q_;
  }


  template <typename T, std::size_t R, std::size_t C>
  constexpr const typename qr_decomposition<T, R, C>::q_t&
  qr_decomposition<T, R, C>::get_q() const
  {
    return q_;
  }


  template <typename T, std::size_t R, std::size_t C>
  constexpr typename qr_decomposition<T, R, C>::r_t&
  qr_decomposition<T, R, C>::get_r()
  {
    return r_;
  }


  template <typename T, std::size_t R, std::size_t C>
  constexpr const typename qr_decomposition<T, R, C>::r_t&
  qr_decomposition<T, R, C>::get_r() const
  {
    return r_;
  }

} // !namespace mfmat
