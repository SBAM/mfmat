#ifndef MFMAT_CT_SEQUENCE_HELPERS_HPP_
# define MFMAT_CT_SEQUENCE_HELPERS_HPP_

# include <utility>

namespace mfmat
{

  /**
   * @tparam MIN range start value
   * @tparam MAX range end value (excluded)
   * @return an index_sequence ranging from MIN to MAX (excluded)
   */
  template <std::size_t MIN, std::size_t MAX>
  constexpr auto make_index_range()
  {
    if constexpr (MIN > MAX)
      return std::make_index_sequence<0>{};
    else
    {
      auto build = []<std::size_t... Is>(std::index_sequence<Is...>)
        {
          return std::index_sequence<MIN + Is...>{};
        };
      return build(std::make_index_sequence<MAX - MIN>{});
    }
  }


  /**
   * @tparam Is first index_sequence
   * @tparam Js second index_sequence
   * @return concatenation of the two given index_sequences
   */
  template <std::size_t... Is, std::size_t... Js>
  constexpr auto cat_index_sequence(std::index_sequence<Is...>,
                                    std::index_sequence<Js...>)
  {
    return std::index_sequence<Is..., Js...>{};
  }


  /**
   * @tparam R matrix's row count
   * @tparam C matrix's column count, defaulted to R
   * @tparam CURR_ROW keeps track of current processed row
   * @return an index_sequence using row direction, pointing to upper
   *         matrix's cells, including its diagonal cells
   */
  template <std::size_t R, std::size_t C = R, std::size_t CURR_ROW = 0>
  constexpr auto make_upper_mat_index_sequence()
  {
    if constexpr (CURR_ROW >= C || CURR_ROW >= R)
      return std::index_sequence<>{};
    else
    {
      constexpr auto start_idx = C * CURR_ROW + CURR_ROW;
      constexpr auto end_idx = C * CURR_ROW + C;
      return cat_index_sequence
        (make_index_range<start_idx, end_idx>(),
         make_upper_mat_index_sequence<R, C, CURR_ROW + 1>());
    }
  }


  /**
   * @tparam R matrix's row count
   * @tparam C matrix's column count, defaulted to R
   * @tparam CURR_ROW keeps track of current processed row
   * @return an index_sequence using row direction, pointing to upper
   *         matrix's cells, excluding its diagonal cells
   */
  template <std::size_t R, std::size_t C = R, std::size_t CURR_ROW = 0>
  constexpr auto make_upper_no_diag_mat_index_sequence()
  {
    if constexpr (CURR_ROW >= C || CURR_ROW >= R)
      return std::index_sequence<>{};
    else
    {
      constexpr auto start_idx = C * CURR_ROW + CURR_ROW + 1;
      constexpr auto end_idx = C * CURR_ROW + C;
      return cat_index_sequence
        (make_index_range<start_idx, end_idx>(),
         make_upper_no_diag_mat_index_sequence<R, C, CURR_ROW + 1>());
    }
  }


  /// @typedef decl_ic shorthand to declare a std::size_t constant
  template <std::size_t IC>
  using decl_ic = std::integral_constant<std::size_t, IC>;

} // !namespace mfmat

#endif // !MFMAT_CT_SEQUENCE_HELPERS_HPP_
