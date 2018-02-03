namespace mfmat
{

  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>::dense_matrix() noexcept
  {
    storage_.fill({});
  }


  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>::dense_matrix(identity_tag) noexcept
  {
    static_assert(R == C, "Identity constructor requires a square matrix");
    storage_.fill({});
    for (std::size_t i = 0; i < R; ++i)
      storage_[i][i] = 1;
  }


  template <typename T, std::size_t R, std::size_t C>
  constexpr dense_matrix<T, R, C>::dense_matrix(matrix_init_list<T> mil) noexcept
  {
    assert(mil.size() == R);
    std::size_t i = 0;
    for (auto ril : mil)
    {
      assert(ril.size() == C);
      std::size_t j = 0;
      for (auto c : ril)
        storage_[i][j++] = c;
      ++i;
    }
  }


  template <typename T, std::size_t R, std::size_t C>
  dense_matrix<T, R, C>::dense_matrix(column_init_list<T> cil) noexcept
  {
    assert(cil.size() == R);
    static_assert(C == 1, "Requires a vector");
    std::size_t i = 0;
    for (auto c : cil)
      storage_[i++][0] = c;
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

} // !namespace mfmat
