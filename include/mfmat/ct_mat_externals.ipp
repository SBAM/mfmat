namespace mfmat
{

  /**
   * @defgroup ExternalFunctions External functions implementation
   * @{
   */
  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C> operator+(ct_mat<T, R, C> lhs, T rhs) noexcept
  {
    lhs += rhs;
    return lhs;
  }


  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C> operator+(T lhs, ct_mat<T, R, C> rhs) noexcept
  {
    return rhs + lhs;
  }


  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C> operator-(ct_mat<T, R, C> lhs, T rhs) noexcept
  {
    lhs -= rhs;
    return lhs;
  }


  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C> operator*(ct_mat<T, R, C> lhs, T rhs) noexcept
  {
    lhs *= rhs;
    return lhs;
  }


  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C> operator*(T lhs, ct_mat<T, R, C> rhs) noexcept
  {
    return rhs * lhs;
  }


  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C> operator/(ct_mat<T, R, C> lhs, T rhs) noexcept
  {
    lhs /= rhs;
    return lhs;
  }


  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C>
  operator+(ct_mat<T, R, C> lhs, const ct_mat<T, R, C>& rhs) noexcept
  {
    lhs += rhs;
    return lhs;
  }


  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C>
  operator-(ct_mat<T, R, C> lhs, const ct_mat<T, R, C>& rhs) noexcept
  {
    lhs -= rhs;
    return lhs;
  }


  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, C, R> transpose(const ct_mat<T, R, C>& arg) noexcept
  {
    auto res = ct_mat<T, C, R>();
    auto cell_swap_copy = [&]<std::size_t... Is>(std::index_sequence<Is...>)
      {
        ((res.template scan<op_way::row, Is>() =
          arg.template scan<op_way::col, Is>()), ...);
      };
    cell_swap_copy(std::make_index_sequence<R * C>{});
    return res;
  }


  template <op_way OW1, std::size_t IDX1,
            op_way OW2, std::size_t IDX2,
            typename M1, typename M2>
  auto dot(const M1& mat1, const M2& mat2) noexcept
  {
    constexpr auto L1 = OW1 == op_way::row ? M1::col_count : M1::row_count;
    constexpr auto L2 = OW2 == op_way::row ? M2::col_count : M2::row_count;
    static_assert(L1 == L2, "Incompatible vectors lengths");
    auto sub_dot = [&]<std::size_t... Is>(std::index_sequence<Is...>)
      {
        return ((mat1.template get<OW1, IDX1, Is>() *
                 mat2.template get<OW2, IDX2, Is>()) + ...);
      };
    return sub_dot(std::make_index_sequence<L1>{});
  }


  template <typename M1, typename M2>
  auto operator*(const M1& lhs, const M2& rhs) noexcept
  {
    static_assert(M1::col_count == M2::row_count,
                  "Incompatible matrices, cannot multiply");
    using RES_T = decltype(typename M1::cell_t{} * typename M2::cell_t{});
    auto res = ct_mat<RES_T, M1::row_count, M2::col_count>();
    auto sub_mul = [&]<std::size_t... Is>(std::index_sequence<Is...>)
      {
        ((res.template scan<op_way::row, Is>() =
            dot<op_way::row, Is / M2::col_count,
                op_way::col, Is % M2::col_count>(lhs, rhs)), ...);
      };
    sub_mul(std::make_index_sequence<M1::row_count * M2::col_count>{});
    return res;
  }
  /// @}

} // !namespace mfmat
