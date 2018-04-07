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


  template <op_way OW_SRC, std::size_t IDX_SRC,
            op_way OW_DST, std::size_t IDX_DST,
            typename M_SRC, typename M_DST>
  void copy_vector(const M_SRC& src, M_DST& dst) noexcept
  {
    constexpr auto LS = OW_SRC == op_way::row ?
      M_SRC::col_count : M_SRC::row_count;
    constexpr auto LD = OW_DST == op_way::row ?
      M_DST::col_count : M_DST::row_count;
    static_assert(LS == LD, "Incompatible vectors lengths");
    auto sub_copy = [&]<auto... Is>(std::index_sequence<Is...>)
      {
        return ((dst.template get<OW_DST, IDX_DST, Is>() =
                 src.template get<OW_SRC, IDX_SRC, Is>()), ...);
      };
    sub_copy(std::make_index_sequence<LS>{});
  }


  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, C, R> transpose(const ct_mat<T, R, C>& arg) noexcept
  {
    auto res = ct_mat<T, C, R>();
    auto cell_swap_copy = [&]<auto... Is>(std::index_sequence<Is...>)
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
    auto sub_dot = [&]<auto... Is>(std::index_sequence<Is...>)
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
    auto sub_mul = [&]<auto... Is>(std::index_sequence<Is...>)
      {
        ((res.template scan<op_way::row, Is>() =
            dot<op_way::row, Is / M2::col_count,
                op_way::col, Is % M2::col_count>(lhs, rhs)), ...);
      };
    sub_mul(std::make_index_sequence<M1::row_count * M2::col_count>{});
    return res;
  }


  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, 1, C> mean(const ct_mat<T, R, C>& arg) noexcept
  {
    auto res = ct_mat<T, 1, C>();
    // computes mean of column whose index is N
    auto vec_mean = [&]<auto N, auto... Is>
      (decl_ic<N>, std::index_sequence<Is...>)
      {
        return (arg.template get<Is, N>() + ...) / T{R};
      };
    // computes mean on all columns
    auto process_col = [&]<auto... Is>(std::index_sequence<Is...>)
      {
        ((res.template get<0, Is>() =
          vec_mean(decl_ic<Is>{}, std::make_index_sequence<R>{})), ...);
      };
    process_col(std::make_index_sequence<C>{});
    return res;
  }


  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, 1, C>
  std_dev(const ct_mat<T, R, C>& arg,
          const ct_mat_opt<T, 1, C>& pc_mean) noexcept
  {
    auto res = ct_mat<T, 1, C>();
    // computes mean if precomputed mean is not available
    const auto& mean_vec = pc_mean ? *pc_mean : mean(arg);
    // computes standard deviation of column whose index is N
    auto stddev_col = [&]<auto N, auto... Is>
      (decl_ic<N>, std::index_sequence<Is...>)
      {
        return std::sqrt
          (((std::pow(arg.template get<Is, N>() -
                      mean_vec.template get<0, N>(), 2)) + ...) / T{R});
      };
    // computes standard deviation on all columns
    auto process_col = [&]<auto... Is>(std::index_sequence<Is...>)
      {
        ((res.template get<0, Is>() =
          stddev_col(decl_ic<Is>{}, std::make_index_sequence<R>{})), ...);
      };
    process_col(std::make_index_sequence<C>{});
    return res;
  }


  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, C, C>
  covariance(ct_mat<T, R, C> arg,
             const ct_mat_opt<T, 1, C>& pc_mean) noexcept
  {
    auto res = ct_mat<T, C, C>();
    // center in place arg
    arg.mean_center(pc_mean);
    // 1/num_obs * transpose(centered_arg) * centered_arg
    auto self_mul = [&]<auto... Is>(std::index_sequence<Is...>)
      {
        ((res.template scan<op_way::row, Is>() =
          res.template scan<op_way::col, Is>() =
            dot<op_way::col, Is / C,
                op_way::col, Is % C>(arg, arg) / T{R}), ...);
      };
    self_mul(make_upper_mat_index_sequence<C>());
    return res;
  }
  /// @}

} // !namespace mfmat
