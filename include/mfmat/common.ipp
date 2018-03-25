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
        // scale machine epsilon to values magnitude
        std::numeric_limits<T>::epsilon() *
        std::max(T(1.0), std::abs(lhs) + std::abs(rhs));
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
