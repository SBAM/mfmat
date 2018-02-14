namespace mfmat
{

  /**
   * @defgroup MemberMethods Member methods implementation
   * @{
   */
  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>::dense_matrix() noexcept :
    storage_{{}}
  {
  }


  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>::dense_matrix(identity_tag) noexcept :
    storage_{{}}
  {
    static_assert(R == C, "Identity constructor requires a square matrix");
    auto diag_set = [this]<std::size_t... Is>(std::index_sequence<Is...>)
      {
        ((get<Is, Is>() = 1), ...);
      };
    using ISEQ = std::make_index_sequence<R>;
    diag_set(ISEQ{});
  }


  template <typename T, std::size_t R, std::size_t C>
  template <typename T2, std::size_t R2, std::size_t C2>
  dense_matrix<T, R, C>::dense_matrix(const T2(&mil)[R2][C2]) noexcept
  {
    static_assert(R == R2, "Rows count mismatch");
    static_assert(C == C2, "Columns count mismatch");
    for (std::size_t i = 0; i < R2; ++i)
      for (std::size_t j = 0; j < C2; ++j)
        storage_[i][j] = mil[i][j];
  }


  template <typename T, std::size_t R, std::size_t C>
  constexpr const T&
  dense_matrix<T, R, C>::operator[](indices idx) const noexcept
  {
    return storage_[idx.first][idx.second];
  }


  template <typename T, std::size_t R, std::size_t C>
  template <std::size_t I, std::size_t J>
  constexpr const T&
  dense_matrix<T, R, C>::get() const noexcept
  {
    return std::get<J>(std::get<I>(storage_));
  }


  template <typename T, std::size_t R, std::size_t C>
  constexpr T&
  dense_matrix<T, R, C>::operator[](indices idx) noexcept
  {
    return storage_[idx.first][idx.second];
  }


  template <typename T, std::size_t R, std::size_t C>
  template <std::size_t I, std::size_t J>
  constexpr T&
  dense_matrix<T, R, C>::get() noexcept
  {
    return std::get<J>(std::get<I>(storage_));
  }


  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>&
  dense_matrix<T, R, C>::operator+=(T val) noexcept
  {
    for (std::size_t i = 0; i < R; ++i)
      for (std::size_t j = 0; j < C; ++j)
        storage_[i][j] += val;
    return *this;
  }


  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>&
  dense_matrix<T, R, C>::operator-=(T val) noexcept
  {
    for (std::size_t i = 0; i < R; ++i)
      for (std::size_t j = 0; j < C; ++j)
        storage_[i][j] -= val;
    return *this;
  }


  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>&
  dense_matrix<T, R, C>::operator*=(T val) noexcept
  {
    for (std::size_t i = 0; i < R; ++i)
      for (std::size_t j = 0; j < C; ++j)
        storage_[i][j] *= val;
    return *this;
  }


  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>&
  dense_matrix<T, R, C>::operator/=(T val) noexcept
  {
    for (std::size_t i = 0; i < R; ++i)
      for (std::size_t j = 0; j < C; ++j)
        storage_[i][j] /= val;
    return *this;
  }


  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>&
  dense_matrix<T, R, C>::operator+=(const dense_matrix<T, R, C>& rhs) noexcept
  {
    for (std::size_t i = 0; i < R; ++i)
      for (std::size_t j = 0; j < C; ++j)
        storage_[i][j] += rhs.storage_[i][j];
    return *this;
  }


  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>&
  dense_matrix<T, R, C>::operator-=(const dense_matrix<T, R, C>& rhs) noexcept
  {
    for (std::size_t i = 0; i < R; ++i)
      for (std::size_t j = 0; j < C; ++j)
        storage_[i][j] -= rhs.storage_[i][j];
    return *this;
  }


  template <typename T, std::size_t R, std::size_t C>
  template <typename T2, std::size_t R2, std::size_t C2>
  auto dense_matrix<T, R, C>::operator*
  (const dense_matrix<T2, R2, C2>& rhs) const noexcept
  {
    static_assert(C == R2, "Incompatible matrices, cannot multiply");
    using RES_T = decltype(T{} * T2{});
    auto res = dense_matrix<RES_T, R, C2>();
    for (std::size_t i = 0; i < R; ++i)
      for (std::size_t j = 0; j < C2; ++j)
        for (std::size_t k = 0; k < C; ++k)
          res.storage_[i][j] += storage_[i][k] * rhs.storage_[k][j];
    return res;
  }


  template <typename T, std::size_t R, std::size_t C>
  bool dense_matrix<T, R, C>::operator==
  (const dense_matrix<T, R, C>& rhs) const noexcept
  {
    for (std::size_t i = 0; i < R; ++i)
      for (std::size_t j = 0; j < C; ++j)
        if (!are_equal(storage_[i][j],rhs.storage_[i][j]))
          return false;
    return true;
  }


  template <typename T, std::size_t R, std::size_t C>
  bool dense_matrix<T, R, C>::operator!=
  (const dense_matrix<T, R, C>& rhs) const noexcept
  {
    return !operator==(rhs);
  }


  template <typename T, std::size_t R, std::size_t C>
  auto dense_matrix<T, R, C>::transpose() const noexcept
  {
    auto res = dense_matrix<T, C, R>();
    for (std::size_t i = 0; i < R; ++i)
      for (std::size_t j = 0; j < C; ++j)
        res.storage_[j][i] = storage_[i][j];
    return res;
  }


  template <typename T, std::size_t R, std::size_t C>
  constexpr T dense_matrix<T, R, C>::trace() const noexcept
  {
    static_assert(R == C, "Trace only applies to a square matrix");
    auto diag_sum = [this]<std::size_t... Is>(std::index_sequence<Is...>)
      {
        return (get<Is, Is>() + ... + T{});
      };
    using ISEQ = std::make_index_sequence<R>;
    return diag_sum(ISEQ{});
  }


  template <typename T, std::size_t R, std::size_t C>
  constexpr T dense_matrix<T, R, C>::rec_det() const noexcept
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
          auto res = dense_matrix<T, R - 1, C - 1>();
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
  /// @}

} // !namespace mfmat
