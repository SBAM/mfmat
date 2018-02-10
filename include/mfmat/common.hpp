#ifndef MFMAT_COMMON_HPP_
# define MFMAT_COMMON_HPP_

# include <cstdint>
# include <utility>

namespace mfmat
{

  /// @brief Identity initialization tag
  struct identity_tag {};

  /// @brief Specifies a matrix cell's indices (first=row_pos, second=col_pos)
  using indices = std::pair<std::size_t, std::size_t>;

} // !namespace mfmat

#endif // !MFMAT_COMMON_HPP_
