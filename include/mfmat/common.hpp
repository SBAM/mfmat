#ifndef MFMAT_COMMON_HPP_
# define MFMAT_COMMON_HPP_

# include <cstdint>
# include <initializer_list>
# include <utility>

namespace mfmat
{

  /// @brief Identity initialization tag
  struct identity_tag {};


  /// @typedef matrix_init_list Shorthand to full matrix initialization list
  template <typename T>
  using matrix_init_list = std::initializer_list<std::initializer_list<T>>;


  /// @typedef column_init_list Shorthand to single column initialization list
  template <typename T>
  using column_init_list = std::initializer_list<T>;


  /// @brief Specifies a matrix cell's indices (first=row_pos, second=col_pos)
  using indices = std::pair<std::size_t, std::size_t>;

} // !namespace mfmat

#endif // !MFMAT_COMMON_HPP_
