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
  /// @}

} // !namespace mfmat
