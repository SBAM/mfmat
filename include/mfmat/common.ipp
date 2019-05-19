namespace mfmat
{

  template <typename T>
  constexpr bool are_equal(T lhs, T rhs) noexcept
  {
    if constexpr (std::is_floating_point_v<T>)
      return
        std::abs(lhs - rhs) <=
        // scale machine epsilon to values magnitude
        std::numeric_limits<T>::epsilon() *
        std::max(T(1.0), std::abs(lhs) + std::abs(rhs));
    else
      return lhs == rhs;
  }


  template <typename T>
  constexpr bool is_zero(T arg) noexcept
  {
    if constexpr (std::is_floating_point_v<T>)
      return std::abs(arg) <= std::numeric_limits<T>::epsilon();
    else
      return arg == T{};
  }


  template <typename T, typename>
  constexpr bool is_zero(T arg, T eps_multiplier) noexcept
  {
    return std::abs(arg) <= eps_multiplier * std::numeric_limits<T>::epsilon();
  }

} // !namespace mfmat
