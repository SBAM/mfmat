namespace mfmat
{

  /**
   * @defgroup MemberMethods Member methods implementation
   * @{
   */
  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C>::ct_mat() noexcept :
    storage_{{}}
  {
  }


  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C>::ct_mat(identity_tag) noexcept :
    storage_{{}}
  {
    static_assert(R == C, "Identity constructor requires a square matrix");
    auto diag_set = [this]<auto... Is>(std::index_sequence<Is...>)
      {
        ((get<Is, Is>() = 1), ...);
      };
    diag_set(std::make_index_sequence<R>{});
  }


  template <typename T, std::size_t R, std::size_t C>
  template <typename T2, std::size_t R2, std::size_t C2>
  ct_mat<T, R, C>::ct_mat(const T2(&mil)[R2][C2]) noexcept
  {
    static_assert(R == R2, "Rows count mismatch");
    static_assert(C == C2, "Columns count mismatch");
    constexpr auto getter = [](const T2(&mil)[R2][C2], std::size_t idx)
      {
        return mil[idx / C2][idx % C2];
      };
    auto cell_copy = [&]<auto... Is>(std::index_sequence<Is...>)
      {
        ((this->scan<op_way::row, Is>() = getter(mil, Is)), ...);
      };
    cell_copy(std::make_index_sequence<R * C>{});
  }


  template <typename T, std::size_t R, std::size_t C>
  template <std::size_t R2, std::size_t C2, std::size_t... IDXs>
  ct_mat<T, R, C>::ct_mat(const ct_mat<T, R2, C2>& rhs,
                          std::index_sequence<IDXs...> seq) noexcept :
    storage_{{}}
  {
    static_assert(seq.size() <= R * C,
                  "Sequence is larger than this matrix's cell count");
    static_assert(seq_max(seq).value < R2 * C2,
                  "Sequence max index points outside rhs");
    auto cell_copy = [&]<std::size_t I, std::size_t HEAD, std::size_t... TAIL>
      (const auto& self, decl_ic<I>, std::index_sequence<HEAD, TAIL...>)
      {
        this->scan<op_way::row, I>() = rhs.template scan<op_way::row, HEAD>();
        if constexpr(sizeof...(TAIL) > 0)
          self(self, decl_ic<I + 1>{}, std::index_sequence<TAIL...>{});
      };
    cell_copy(cell_copy, decl_ic<0>{}, seq);
  }


  template <typename T, std::size_t R, std::size_t C>
  T ct_mat<T, R, C>::operator[](indices idx) const noexcept
  {
    return storage_[idx.first][idx.second];
  }


  template <typename T, std::size_t R, std::size_t C>
  template <std::size_t R_IDX, std::size_t C_IDX>
  constexpr T ct_mat<T, R, C>::get() const
  {
    return std::get<C_IDX>(std::get<R_IDX>(storage_));
  }


  template <typename T, std::size_t R, std::size_t C>
  template <op_way OW, std::size_t IDX1, std::size_t IDX2>
  constexpr T ct_mat<T, R, C>::get() const
  {
    if constexpr (OW == op_way::row)
      return std::get<IDX2>(std::get<IDX1>(storage_));
    else
      return std::get<IDX1>(std::get<IDX2>(storage_));
  }


  template <typename T, std::size_t R, std::size_t C>
  template <op_way OW, std::size_t IDX>
  constexpr T ct_mat<T, R, C>::scan() const
  {
    if constexpr (OW == op_way::row)
      return std::get<IDX % C>(std::get<IDX / C>(storage_));
    else
      return std::get<IDX / R>(std::get<IDX % R>(storage_));
  }


  template <typename T, std::size_t R, std::size_t C>
  T& ct_mat<T, R, C>::operator[](indices idx) noexcept
  {
    return storage_[idx.first][idx.second];
  }


  template <typename T, std::size_t R, std::size_t C>
  template <std::size_t R_IDX, std::size_t C_IDX>
  constexpr T& ct_mat<T, R, C>::get()
  {
    return std::get<C_IDX>(std::get<R_IDX>(storage_));
  }


  template <typename T, std::size_t R, std::size_t C>
  template <op_way OW, std::size_t IDX1, std::size_t IDX2>
  constexpr T& ct_mat<T, R, C>::get()
  {
    if constexpr (OW == op_way::row)
      return std::get<IDX2>(std::get<IDX1>(storage_));
    else
      return std::get<IDX1>(std::get<IDX2>(storage_));
  }


  template <typename T, std::size_t R, std::size_t C>
  template <op_way OW, std::size_t IDX>
  constexpr T& ct_mat<T, R, C>::scan()
  {
    if constexpr (OW == op_way::row)
      return std::get<IDX % C>(std::get<IDX / C>(storage_));
    else
      return std::get<IDX / R>(std::get<IDX % R>(storage_));
  }


  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C>& ct_mat<T, R, C>::operator+=(T val) noexcept
  {
    auto cell_add = [=]<auto... Is>(std::index_sequence<Is...>)
      {
        ((this->scan<op_way::row, Is>() += val), ...);
      };
    cell_add(std::make_index_sequence<R * C>{});
    return *this;
  }


  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C>& ct_mat<T, R, C>::operator-=(T val) noexcept
  {
    auto cell_sub = [=]<auto... Is>(std::index_sequence<Is...>)
      {
        ((this->scan<op_way::row, Is>() -= val), ...);
      };
    cell_sub(std::make_index_sequence<R * C>{});
    return *this;
  }


  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C>& ct_mat<T, R, C>::operator*=(T val) noexcept
  {
    auto cell_mul = [=]<auto... Is>(std::index_sequence<Is...>)
      {
        ((this->scan<op_way::row, Is>() *= val), ...);
      };
    cell_mul(std::make_index_sequence<R * C>{});
    return *this;
  }


  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C>& ct_mat<T, R, C>::operator/=(T val) noexcept
  {
    auto cell_div = [=]<auto... Is>(std::index_sequence<Is...>)
      {
        ((this->scan<op_way::row, Is>() /= val), ...);
      };
    cell_div(std::make_index_sequence<R * C>{});
    return *this;
  }


  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C>&
  ct_mat<T, R, C>::operator+=(const ct_mat<T, R, C>& rhs) noexcept
  {
    auto cell_add = [&]<auto... Is>(std::index_sequence<Is...>)
      {
        ((this->scan<op_way::row, Is>() +=
          rhs.template scan<op_way::row, Is>()), ...);
      };
    cell_add(std::make_index_sequence<R * C>{});
    return *this;
  }


  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C>&
  ct_mat<T, R, C>::operator-=(const ct_mat<T, R, C>& rhs) noexcept
  {
    auto cell_sub = [&]<auto... Is>(std::index_sequence<Is...>)
      {
        ((this->scan<op_way::row, Is>() -=
          rhs.template scan<op_way::row, Is>()), ...);
      };
    cell_sub(std::make_index_sequence<R * C>{});
    return *this;
  }


  template <typename T, std::size_t R, std::size_t C>
  bool ct_mat<T, R, C>::operator==(const ct_mat<T, R, C>& rhs) const noexcept
  {
    auto cell_eq = [&]<auto... Is>(std::index_sequence<Is...>)
      {
        return (are_equal(this->scan<op_way::row, Is>(),
                          rhs.template scan<op_way::row, Is>()) && ...);
      };
    return cell_eq(std::make_index_sequence<R * C>{});
  }


  template <typename T, std::size_t R, std::size_t C>
  bool ct_mat<T, R, C>::operator!=(const ct_mat<T, R, C>& rhs) const noexcept
  {
    return !operator==(rhs);
  }


  template <typename T, std::size_t R, std::size_t C>
  template <op_way OW, std::size_t IDX>
  T ct_mat<T, R, C>::norm() const noexcept
  {
    constexpr auto VL = OW == op_way::row ? C : R;
    auto sum_square = [this]<auto... Is>(std::index_sequence<Is...>)
      {
        return ((this->get<OW, IDX, Is>() * this->get<OW, IDX, Is>()) + ...);
      };
    return std::sqrt(sum_square(std::make_index_sequence<VL>{}));
  }


  template <typename T, std::size_t R, std::size_t C>
  template <op_way OW, std::size_t IDX>
  bool ct_mat<T, R, C>::normalize() noexcept
  {
    if (auto vec_norm = norm<OW, IDX>(); is_zero(vec_norm))
      return false;
    else
    {
      constexpr auto VL = OW == op_way::row ? C : R;
      auto cell_div = [&]<auto... Is>(std::index_sequence<Is...>)
      {
        return ((this->get<OW, IDX, Is>() /= vec_norm), ...);
      };
      cell_div(std::make_index_sequence<VL>{});
      return true;
    }
  }


  template <typename T, std::size_t R, std::size_t C>
  void ct_mat<T, R, C>::transpose() noexcept
  {
    static_assert(R == C, "In place transpose only applies to a square matrix");
    auto cell_swap = [&]<auto... Is>(std::index_sequence<Is...>)
      {
        ((std::swap(this->scan<op_way::row, Is>(),
                    this->scan<op_way::col, Is>())), ...);
      };
    cell_swap(make_upper_no_diag_mat_index_sequence<R, C>());
  }


  template <typename T, std::size_t R, std::size_t C>
  T ct_mat<T, R, C>::trace() const noexcept
  {
    static_assert(R == C, "Trace only applies to a square matrix");
    auto diag_sum = [this]<auto... Is>(std::index_sequence<Is...>)
      {
        return (get<Is, Is>() + ...);
      };
    return diag_sum(std::make_index_sequence<R>{});
  }


  template <typename T, std::size_t R, std::size_t C>
  T ct_mat<T, R, C>::rec_det() const noexcept
  {
    static_assert(R == C, "Determinant only applies to a square matrix");
    static_assert(R > 1, "Determinant requires at least a 2x2 matrix");
    if constexpr (R == 2)
      return get<0, 0>() * get<1, 1>() - get<0, 1>() * get<1, 0>();
    else
    {
      // extracts sub matrix excluding first row and excluding specified column
      auto extract_sub = [this]<std::size_t EC>(decl_ic<EC>)
        {
          constexpr auto s0 = std::make_index_sequence<R * C>{};
          constexpr auto s1 = remove_seq<op_way::row, R, C, 0>(s0);
          constexpr auto s2 = remove_seq<op_way::col, R, C, EC>(s1);
          return ct_mat<T, R - 1, C - 1>(*this, s2);
        };
      auto process_cell = [&]<std::size_t I>(decl_ic<I>)
        {
          if (is_zero(this->get<0, I>()))
            return T{};
          else
          {
            if constexpr(I % 2 == 0)
              return this->get<0, I>() * extract_sub(decl_ic<I>{}).rec_det();
            else
              return -this->get<0, I>() * extract_sub(decl_ic<I>{}).rec_det();
          }
        };
      auto scan_col = [&]<std::size_t... Cs>(std::index_sequence<Cs...>)
        {
          return (process_cell(decl_ic<Cs>{}) + ...);
        };
      return scan_col(std::make_index_sequence<C>{});
    }
  }
  /// @}

} // !namespace mfmat
