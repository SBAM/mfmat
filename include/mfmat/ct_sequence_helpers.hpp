#ifndef MFMAT_CT_SEQUENCE_HELPERS_HPP_
# define MFMAT_CT_SEQUENCE_HELPERS_HPP_

# include "common.hpp"

namespace mfmat
{

  /// @typedef decl_ic shorthand to declare an intergral constant
  template <std::size_t IC>
  using decl_ic = std::integral_constant<std::size_t, IC>;


  /**
   * @tparam MIN range start value
   * @tparam MAX range end value (excluded)
   * @return an index_sequence ranging from MIN to MAX (excluded)
   */
  template <std::size_t MIN, std::size_t MAX>
  constexpr auto make_index_range()
  {
    if constexpr (MIN > MAX)
      return std::index_sequence<>{};
    else
    {
      constexpr auto build = []<std::size_t... Is>(std::index_sequence<Is...>)
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


  /**
   * @tparam R matrix's row count
   * @tparam C matrix's column count, defaulted to R
   * @tparam CURR_ROW keeps track of current processed row
   * @return an index_sequence using row direction, pointing to all
   *         matrix's cells, excluding its diagonal cells
   */
  template <std::size_t R, std::size_t C = R, std::size_t CURR_ROW = 0>
  constexpr auto make_no_diag_index_sequence()
  {
    if constexpr (CURR_ROW >= R)
      return std::index_sequence<>{};
    else
    {
      constexpr auto start_idx = C * CURR_ROW;
      constexpr auto diag_idx = start_idx + CURR_ROW;
      constexpr auto end_idx = start_idx + C;
      if constexpr (diag_idx > end_idx)
        return cat_index_sequence
          (make_index_range<start_idx, end_idx>(),
           make_no_diag_index_sequence<R, C, CURR_ROW + 1>());
      else
        return cat_index_sequence
          (cat_index_sequence(make_index_range<start_idx, diag_idx>(),
                              make_index_range<diag_idx + 1, end_idx>()),
           make_no_diag_index_sequence<R, C, CURR_ROW + 1>());
    }
  }


  /**
   * @brief This helper takes an index_sequence running along rows of a matrix
   *        whose dimensions are R=row/C=col and excludes indices that belong
   *        to the REMnth Row/Col (as specified by OW)
   * @tparam OW specifies if a row or column is to be removed
   * @tparam R matrix's row count
   * @tparam C matrix's column count
   * @tparam REM identifies which row/column is to be removed
   * @tparam Is input index sequence to filter
   * @return a filtered index sequence
   */
  template <op_way OW, std::size_t R, std::size_t C,
            std::size_t REM, std::size_t... Is>
  constexpr auto remove_seq(std::index_sequence<Is...> seq)
  {
    if constexpr(OW == op_way::row)
      static_assert(REM < R, "Row index is greater than row count");
    else
      static_assert(REM < C, "Column index is greater than column count");
    if constexpr(seq.size() == 0)
      return std::index_sequence<>{};
    else
    {
      // silence warning, self is not used when TAIL is empty
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-but-set-parameter"
      constexpr auto build = []<std::size_t HEAD, std::size_t... TAIL>
        (const auto& self, std::index_sequence<HEAD, TAIL...>)
        {
          constexpr bool filter = (OW == op_way::row && HEAD / C == REM) ||
                                  (OW == op_way::col && HEAD % C == REM);
          if constexpr(sizeof...(TAIL) == 0)
          {
            if constexpr (filter)
              return std::index_sequence<>{};
            else
              return std::index_sequence<HEAD>{};
          }
          else
          {
            if constexpr (filter)
              return self(self, std::index_sequence<TAIL...>{});
            else
              return cat_index_sequence
                (std::index_sequence<HEAD>{},
                 self(self, std::index_sequence<TAIL...>{}));
          }
        };
      return build(build, seq);
# pragma GCC diagnostic pop
    }
  }


  /**
   * @tparam HEAD first index sequence element
   * @tparam TAIL remaining index sequence elements past HEAD
   * @tparam MAX current sequence maximum
   * @return max value from an input index sequence
   */
  template <std::size_t HEAD, std::size_t... TAIL, std::size_t MAX = 0>
  constexpr auto seq_max(std::index_sequence<HEAD, TAIL...>, decl_ic<MAX> = {})
  {
    if constexpr(sizeof...(TAIL) == 0)
      if constexpr(HEAD > MAX)
        return decl_ic<HEAD>{};
      else
        return decl_ic<MAX>{};
    else
      if constexpr(HEAD > MAX)
        return seq_max(std::index_sequence<TAIL...>{}, decl_ic<HEAD>{});
      else
        return seq_max(std::index_sequence<TAIL...>{}, decl_ic<MAX>{});
  }

} // !namespace mfmat

#endif // !MFMAT_CT_SEQUENCE_HELPERS_HPP_
