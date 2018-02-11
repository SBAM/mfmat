namespace mfmat
{

  template <typename T>
  constexpr bool are_equal(T lhs, T rhs) noexcept
  {
    if constexpr (std::numeric_limits<T>::is_integer)
      return lhs == rhs;
    else
      return
        std::abs(lhs - rhs) <=
        // scale machine epsilon to values magnitude (precision of 1 ULP)
        std::numeric_limits<T>::epsilon() * std::abs(lhs + rhs);
  }


  template <typename T>
  constexpr bool is_zero(T arg) noexcept
  {
    if constexpr (std::numeric_limits<T>::is_integer)
      return arg == T{};
    else
      return std::abs(arg) <= std::numeric_limits<T>::epsilon();
  }

} // !namespace mfmat
