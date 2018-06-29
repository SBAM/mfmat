#ifndef MFMAT_COMMON_HPP_
# define MFMAT_COMMON_HPP_

# include <cassert>
# include <cstdint>
# include <cmath>
# include <limits>
# include <utility>

namespace mfmat
{

  /// @brief Identity initialization tag
  struct identity_tag
  {
    explicit identity_tag() = default;
  };

  /// @brief Specifies a matrix cell's indices (first=row_pos, second=col_pos)
  using indices = std::pair<std::size_t, std::size_t>;

  /// @brief compares for equality regardless of T (int/real)
  template <typename T>
  constexpr bool are_equal(T lhs, T rhs) noexcept;

  /// @brief compares against zero regardless of T (int/real)
  template <typename T>
  constexpr bool is_zero(T arg) noexcept;

  /// @brief compares floating point type against zero with tolerance
  template <typename T,
            typename = std::enable_if_t<std::is_floating_point_v<T>>>
  constexpr bool is_zero(T arg, T eps_multiplier) noexcept;

  /// @brief defines operation way, row or column wise
  enum class op_way
  {
    row,
    col
  };

} // !namespace mfmat

# include "common.ipp"

#endif // !MFMAT_COMMON_HPP_
