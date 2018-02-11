namespace mfmat
{

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
    for (std::size_t i = 0; i < R; ++i)
      storage_[i][i] = 1;
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
    if constexpr (std::numeric_limits<T>::is_integer)
    {
      for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j)
          if (storage_[i][j] != rhs.storage_[i][j])
            return false;
      return true;
    }
    else
    {
      for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j)
        {
          auto lhs_cell = storage_[i][j];
          auto rhs_cell = rhs.storage_[i][j];
          if (std::abs(lhs_cell - rhs_cell) >
              // scale machine epsilon to values magnitude (precision of 1 ULP)
              std::numeric_limits<T>::epsilon() * std::abs(lhs_cell + rhs_cell))
            return false;
        }
      return true;
    }
  }


  template <typename T, std::size_t R, std::size_t C>
  bool dense_matrix<T, R, C>::operator!=
  (const dense_matrix<T, R, C>& rhs) const noexcept
  {
    return !operator==(rhs);
  }


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

} // !namespace mfmat
