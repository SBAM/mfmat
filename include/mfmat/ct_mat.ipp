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
    auto diag_set = [this]<std::size_t... Is>(std::index_sequence<Is...>)
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
    auto cell_copy = [&]<std::size_t... Is>(std::index_sequence<Is...>)
      {
        ((this->scan<op_way::row, Is>() = getter(mil, Is)), ...);
      };
    cell_copy(std::make_index_sequence<R * C>{});
  }


  template <typename T, std::size_t R, std::size_t C>
  constexpr T ct_mat<T, R, C>::operator[](indices idx) const noexcept
  {
    return storage_[idx.first][idx.second];
  }


  template <typename T, std::size_t R, std::size_t C>
  template <std::size_t R_IDX, std::size_t C_IDX>
  constexpr T ct_mat<T, R, C>::get() const noexcept
  {
    return std::get<C_IDX>(std::get<R_IDX>(storage_));
  }


  template <typename T, std::size_t R, std::size_t C>
  template <op_way OW, std::size_t IDX1, std::size_t IDX2>
  constexpr T ct_mat<T, R, C>::get() const noexcept
  {
    if constexpr (OW == op_way::row)
      return std::get<IDX2>(std::get<IDX1>(storage_));
    else
      return std::get<IDX1>(std::get<IDX2>(storage_));
  }


  template <typename T, std::size_t R, std::size_t C>
  template <op_way OW, std::size_t IDX>
  constexpr T ct_mat<T, R, C>::scan() const noexcept
  {
    if constexpr (OW == op_way::row)
      return std::get<IDX % C>(std::get<IDX / C>(storage_));
    else
      return std::get<IDX / R>(std::get<IDX % R>(storage_));
  }


  template <typename T, std::size_t R, std::size_t C>
  constexpr T& ct_mat<T, R, C>::operator[](indices idx) noexcept
  {
    return storage_[idx.first][idx.second];
  }


  template <typename T, std::size_t R, std::size_t C>
  template <std::size_t R_IDX, std::size_t C_IDX>
  constexpr T& ct_mat<T, R, C>::get() noexcept
  {
    return std::get<C_IDX>(std::get<R_IDX>(storage_));
  }


  template <typename T, std::size_t R, std::size_t C>
  template <op_way OW, std::size_t IDX1, std::size_t IDX2>
  constexpr T& ct_mat<T, R, C>::get() noexcept
  {
    if constexpr (OW == op_way::row)
      return std::get<IDX2>(std::get<IDX1>(storage_));
    else
      return std::get<IDX1>(std::get<IDX2>(storage_));
  }


  template <typename T, std::size_t R, std::size_t C>
  template <op_way OW, std::size_t IDX>
  constexpr T& ct_mat<T, R, C>::scan() noexcept
  {
    if constexpr (OW == op_way::row)
      return std::get<IDX % C>(std::get<IDX / C>(storage_));
    else
      return std::get<IDX / R>(std::get<IDX % R>(storage_));
  }


  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C>& ct_mat<T, R, C>::operator+=(T val) noexcept
  {
    auto cell_add = [=]<std::size_t... Is>(std::index_sequence<Is...>)
      {
        ((this->scan<op_way::row, Is>() += val), ...);
      };
    cell_add(std::make_index_sequence<R * C>{});
    return *this;
  }


  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C>& ct_mat<T, R, C>::operator-=(T val) noexcept
  {
    auto cell_sub = [=]<std::size_t... Is>(std::index_sequence<Is...>)
      {
        ((this->scan<op_way::row, Is>() -= val), ...);
      };
    cell_sub(std::make_index_sequence<R * C>{});
    return *this;
  }


  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C>& ct_mat<T, R, C>::operator*=(T val) noexcept
  {
    auto cell_mul = [=]<std::size_t... Is>(std::index_sequence<Is...>)
      {
        ((this->scan<op_way::row, Is>() *= val), ...);
      };
    cell_mul(std::make_index_sequence<R * C>{});
    return *this;
  }


  template <typename T, std::size_t R, std::size_t C>
  ct_mat<T, R, C>& ct_mat<T, R, C>::operator/=(T val) noexcept
  {
    auto cell_div = [=]<std::size_t... Is>(std::index_sequence<Is...>)
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
    auto cell_add = [&]<std::size_t... Is>(std::index_sequence<Is...>)
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
    auto cell_sub = [&]<std::size_t... Is>(std::index_sequence<Is...>)
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
    auto cell_eq = [&]<std::size_t... Is>(std::index_sequence<Is...>)
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
  auto ct_mat<T, R, C>::transpose() const noexcept
  {
    auto res = ct_mat<T, C, R>();
    auto cell_swap_copy = [&]<std::size_t... Is>(std::index_sequence<Is...>)
      {
        ((res.template scan<op_way::row, Is>() =
          this->scan<op_way::col, Is>()), ...);
      };
    cell_swap_copy(std::make_index_sequence<R * C>{});
    return res;
  }


  template <typename T, std::size_t R, std::size_t C>
  constexpr T ct_mat<T, R, C>::trace() const noexcept
  {
    static_assert(R == C, "Trace only applies to a square matrix");
    auto diag_sum = [this]<std::size_t... Is>(std::index_sequence<Is...>)
      {
        return (get<Is, Is>() + ... + T{});
      };
    return diag_sum(std::make_index_sequence<R>{});
  }


  template <typename T, std::size_t R, std::size_t C>
  constexpr T ct_mat<T, R, C>::rec_det() const noexcept
  {
    static_assert(R == C, "Determinant only applies to a square matrix");
    static_assert(R > 1, "Determinant requires at least a 2x2 matrix");
    if constexpr (R == 2)
      return get<0, 0>() * get<1, 1>() - get<0, 1>() * get<1, 0>();
    else
    {
      // extracts sub matrix excluding first row and excluding specified column
      auto sub = [this](std::size_t excluded_column)
        {
          auto res = ct_mat<T, R - 1, C - 1>();
          for (std::size_t i = 1; i < R; ++i)
            for (std::size_t j = 0; j < C; ++j)
              if (j < excluded_column)
                res.storage_[i - 1][j] = storage_[i][j];
              else
                if (j == excluded_column)
                  continue;
                else
                  res.storage_[i - 1][j - 1] = storage_[i][j];
          return res;
        };
      auto res_det = T{};
      bool sign = true;
      for (std::size_t j = 0; j < C; ++j)
      {
        if (!is_zero(storage_[0][j]))
        {
          if (sign)
            res_det += storage_[0][j] * sub(j).rec_det();
          else
            res_det -= storage_[0][j] * sub(j).rec_det();
        }
        sign = !sign;
      }
      return res_det;
    }
  }
  /// @}

} // !namespace mfmat