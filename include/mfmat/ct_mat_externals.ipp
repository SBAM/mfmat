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
  ct_mat<T, R, C> orthonormalize(const ct_mat<T, R, C>& arg) noexcept
  {
    auto res = ct_mat<T, R, C>();
    // substracts scaled column K (col N projected against col K) from column N
    auto substract_scaled = [&]<auto N, auto K, auto... Is>
      (decl_ic<N>, decl_ic<K>, auto scale, std::index_sequence<Is...>)
      {
        ((res.template get<op_way::col, N, Is>() -=
          res.template get<op_way::col, K, Is>() * scale), ...);
      };
    // computes projection of column N against column K then substracts
    // projection to make both columns orthogonal
    auto substract_proj = [&]<auto N, auto K>(decl_ic<N>, decl_ic<K>)
      {
        /**
         * @note we're reusing the currently processed column, not the original
         *       column from arg, this is the stabilized Gram-Schmidt version.
         *       classical version would use rather use:
         *         dot<op_way::col, K, op_way::col, N>(res, arg)
         */
        auto scale = dot<op_way::col, K, op_way::col, N>(res, res);
        substract_scaled(decl_ic<N>{}, decl_ic<K>{}, scale,
                         make_index_range<0, R>());
      };
    // projects column N on each previously orthonormalized columns Ks
    auto all_proj = [&]<auto N, auto... Ks>
      (decl_ic<N>, std::index_sequence<Ks...>)
      {
        ((substract_proj(decl_ic<N>{}, decl_ic<Ks>{})), ...);
      };
    // processes a single column, copies it to output, substracts projections
    // against previously orthogonalized colums then normalizes it
    auto process_column = [&]<auto... Is>(std::index_sequence<Is...>)
      {
        // copy current column from initial matrix to result
        ((copy_vector<op_way::col, Is, op_way::col, Is>(arg, res),
          // make current column orthogonal to previous columns
          all_proj(decl_ic<Is>{}, make_index_range<0, Is>()),
          // normalize result column
          res.template normalize<op_way::col, Is>()), ...);
      };
    process_column(std::make_index_sequence<C>{});
    return res;
  }
  /// @}

} // !namespace mfmat
