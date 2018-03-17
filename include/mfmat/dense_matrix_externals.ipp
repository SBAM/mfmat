namespace mfmat
{

  /**
   * @defgroup ExternalFunctions External functions implementation
   * @{
   */
  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>
  operator+(const dense_matrix<T, R, C>& lhs, T rhs) noexcept
  {
    auto res = lhs;
    res += rhs;
    return res;
  }


  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>
  operator+(T lhs, const dense_matrix<T, R, C>& rhs) noexcept
  {
    return rhs + lhs;
  }


  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>
  operator-(const dense_matrix<T, R, C>& lhs, T rhs) noexcept
  {
    auto res = lhs;
    res -= rhs;
    return res;
  }


  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>
  operator*(const dense_matrix<T, R, C>& lhs, T rhs) noexcept
  {
    auto res = lhs;
    res *= rhs;
    return res;
  }


  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>
  operator*(T lhs, const dense_matrix<T, R, C>& rhs) noexcept
  {
    return rhs * lhs;
  }


  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>
  operator/(const dense_matrix<T, R, C>& lhs, T rhs) noexcept
  {
    auto res = lhs;
    res /= rhs;
    return res;
  }


  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>
  operator+(const dense_matrix<T, R, C>& lhs,
            const dense_matrix<T, R, C>& rhs) noexcept
  {
    auto res = lhs;
    res += rhs;
    return res;
  }


  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>
  operator-(const dense_matrix<T, R, C>& lhs,
            const dense_matrix<T, R, C>& rhs) noexcept
  {
    auto res = lhs;
    res -= rhs;
    return res;
  }


  template <dot_spec DS, std::size_t S1, std::size_t S2,
            typename M1, typename M2>
  auto dot(const M1& mat1, const M2& mat2) noexcept
  {
    if constexpr (DS == dot_spec::row_col)
    {
      static_assert(M1::col_count == M2::row_count, "Incompatible vectors");
      auto sub_dot = [&]<std::size_t... Is>(std::index_sequence<Is...>)
        {
          return
            ((mat1.template get<S1, Is>() * mat2.template get<Is, S2>()) + ...);
        };
      return sub_dot(std::make_index_sequence<M1::col_count>{});
    }
    else if constexpr (DS == dot_spec::row_row)
    {
      static_assert(M1::col_count == M2::col_count, "Incompatible vectors");
      auto sub_dot = [&]<std::size_t... Is>(std::index_sequence<Is...>)
        {
          return
            ((mat1.template get<S1, Is>() * mat2.template get<S2, Is>()) + ...);
        };
      return sub_dot(std::make_index_sequence<M1::col_count>{});
    }
    else if constexpr (DS == dot_spec::col_row)
    {
      static_assert(M1::row_count == M2::col_count, "Incompatible vectors");
      auto sub_dot = [&]<std::size_t... Is>(std::index_sequence<Is...>)
        {
          return
            ((mat1.template get<Is, S1>() * mat2.template get<S2, Is>()) + ...);
        };
      return sub_dot(std::make_index_sequence<M1::row_count>{});
    }
    else // col_col
    {
      static_assert(DS == dot_spec::col_col);
      static_assert(M1::row_count == M2::row_count, "Incompatible vectors");
      auto sub_dot = [&]<std::size_t... Is>(std::index_sequence<Is...>)
        {
          return
            ((mat1.template get<Is, S1>() * mat2.template get<Is, S2>()) + ...);
        };
      return sub_dot(std::make_index_sequence<M1::row_count>{});
    }
  }
  /// @}

} // !namespace mfmat
