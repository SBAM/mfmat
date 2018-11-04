namespace mfmat
{

  template <typename T>
  qr_decomposition<T>::qr_decomposition(const m_t& input)
  {
    this->operator()(input);
  }


  template <typename T>
  qr_decomposition<T>&
  qr_decomposition<T>::operator()(const m_t& input)
  {
    q_ = input;
    q_.orthonormalize();
    // avoids creating temporary transposed q_
    // r_ = transpose(q_) * input;
    if constexpr (is_ct_mat<T>::value)
    {
      auto transpose_mul = [&]<auto... Is>(std::index_sequence<Is...>)
        {
          ((r_.template scan<op_way::row, Is>() =
              dot<op_way::col, Is / r_t::col_count,
                  op_way::col, Is % r_t::col_count>(q_, input)), ...);
        };
      transpose_mul
        (std::make_index_sequence<r_t::row_count * r_t::col_count>{});
    }
    if constexpr (is_cl_mat<T>::value)
    {
      auto q_dat = ro_bind(q_.storage_);
      auto input_dat = ro_bind(input.storage_);
      if (r_.get_row_count() != input.get_col_count() ||
          r_.get_col_count() != input.get_col_count())
        r_ = r_t(input.get_col_count());
      auto r_dat = wo_bind(r_.storage_);
      auto& ker = cl_kernels_store::instance().matrix_transpose_multiply;
      using KT = typename T::cell_t;
      bind_ker<KT>(ker, cl::NDRange(r_.get_row_count(), r_.get_col_count()),
                   q_dat, q_.get_row_count(),
                   input_dat, input.get_col_count(),
                   r_dat);
      bind_res(r_dat, r_.storage_);
    }
    return *this;
  }


  template <typename T>
  constexpr typename qr_decomposition<T>::q_t&
  qr_decomposition<T>::get_q()
  {
    return q_;
  }


  template <typename T>
  constexpr const typename qr_decomposition<T>::q_t&
  qr_decomposition<T>::get_q() const
  {
    return q_;
  }


  template <typename T>
  constexpr typename qr_decomposition<T>::r_t&
  qr_decomposition<T>::get_r()
  {
    return r_;
  }


  template <typename T>
  constexpr const typename qr_decomposition<T>::r_t&
  qr_decomposition<T>::get_r() const
  {
    return r_;
  }

} // !namespace mfmat
