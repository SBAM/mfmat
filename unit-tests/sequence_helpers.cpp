#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mfmat

#include <iostream>
#include <vector>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/ct_sequence_helpers.hpp>

BOOST_AUTO_TEST_SUITE(sequence_helpers_test_suite)

constexpr auto owr = mfmat::op_way::row;
constexpr auto owc = mfmat::op_way::col;

BOOST_AUTO_TEST_CASE(test_bootstrap)
{
  std::vector<std::size_t> res;
  auto test = [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      (res.push_back(Is), ...);
    };
  test(std::make_index_sequence<5>{});
  BOOST_CHECK(res.size() == 5);
  for (std::size_t i = 0; i < 5; ++i)
    BOOST_CHECK(res[i] == i);
}


BOOST_AUTO_TEST_CASE(test_range_1)
{
  std::vector<std::size_t> res;
  auto test = [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      (res.push_back(Is), ...);
    };
  test(mfmat::make_index_range<0, 5>());
  BOOST_CHECK(res.size() == 5);
  for (std::size_t i = 0; i < 5; ++i)
    BOOST_CHECK(res[i] == i);
}


BOOST_AUTO_TEST_CASE(test_range_2)
{
  std::vector<std::size_t> res;
  auto test = [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      (res.push_back(Is), ...);
    };
  test(mfmat::make_index_range<5, 10>());
  BOOST_CHECK(res.size() == 5);
  for (std::size_t i = 0; i < 5; ++i)
    BOOST_CHECK(res[i] == i + 5);
}


BOOST_AUTO_TEST_CASE(test_range_3)
{
  std::vector<std::size_t> res;
  auto test = [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      (res.push_back(Is), ...);
    };
  test(mfmat::make_index_range<2, 2>());
  BOOST_CHECK(res.empty());
}


BOOST_AUTO_TEST_CASE(test_range_4)
{
  std::vector<std::size_t> res;
  auto test = [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      (res.push_back(Is), ...);
    };
  test(mfmat::make_index_range<1, 0>());
  BOOST_CHECK(res.empty());
}


BOOST_AUTO_TEST_CASE(cat_sequences_1)
{
  std::vector<std::size_t> res;
  auto test = [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      (res.push_back(Is), ...);
    };
  auto seq = mfmat::cat_index_sequence(mfmat::make_index_range<2, 4>(),
                                       mfmat::make_index_range<6, 8>());
  test(seq);
  BOOST_CHECK(res.size() == 4);
  for (std::size_t i = 0; i < 2; ++i)
    BOOST_CHECK(res[i] == i + 2);
  for (std::size_t i = 2; i < 4; ++i)
    BOOST_CHECK(res[i] == i + 4);
}


BOOST_AUTO_TEST_CASE(cat_sequences_2)
{
  std::vector<std::size_t> res;
  auto test = [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      (res.push_back(Is), ...);
    };
  auto seq1 = mfmat::cat_index_sequence
    (mfmat::make_index_range<1, 1>(), mfmat::make_index_range<5, 10>());
  auto seq2 = mfmat::cat_index_sequence
    (seq1, mfmat::make_index_range<2, 4>());
  test(seq2);
  BOOST_CHECK(res.size() == 7);
  for (std::size_t i = 0; i < 5; ++i)
    BOOST_CHECK(res[i] == i + 5);
  for (std::size_t i = 5; i < 7; ++i)
    BOOST_CHECK(res[i] == i - 3);
}


BOOST_AUTO_TEST_CASE(upper_mat_sequence_1)
{
  std::vector<std::size_t> res;
  auto test = [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      (res.push_back(Is), ...);
    };
  test(mfmat::make_upper_mat_index_sequence<1>());
  BOOST_CHECK(res.size() == 1);
  BOOST_CHECK(res[0] == 0);
}


BOOST_AUTO_TEST_CASE(upper_mat_sequence_2)
{
  std::vector<std::size_t> res;
  auto test = [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      (res.push_back(Is), ...);
    };
  /*
   * | 0, 1, 2 |
   * | 3, 4, 5 |
   * | 6, 7, 8 |
   */
  test(mfmat::make_upper_mat_index_sequence<3>());
  BOOST_CHECK(res.size() == 6);
  BOOST_CHECK(res[0] == 0);
  BOOST_CHECK(res[1] == 1);
  BOOST_CHECK(res[2] == 2);
  BOOST_CHECK(res[3] == 4);
  BOOST_CHECK(res[4] == 5);
  BOOST_CHECK(res[5] == 8);
}


BOOST_AUTO_TEST_CASE(upper_mat_sequence_3)
{
  std::vector<std::size_t> res;
  auto test = [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      (res.push_back(Is), ...);
    };
  /*
   * |  0,  1,  2,  3,  4 |
   * |  5,  6,  7,  8,  9 |
   * | 10, 11, 12, 13, 14 |
   */
  test(mfmat::make_upper_mat_index_sequence<3, 5>());
  BOOST_CHECK(res.size() == 12);
  BOOST_CHECK(res[ 0] ==  0);
  BOOST_CHECK(res[ 1] ==  1);
  BOOST_CHECK(res[ 2] ==  2);
  BOOST_CHECK(res[ 3] ==  3);
  BOOST_CHECK(res[ 4] ==  4);
  BOOST_CHECK(res[ 5] ==  6);
  BOOST_CHECK(res[ 6] ==  7);
  BOOST_CHECK(res[ 7] ==  8);
  BOOST_CHECK(res[ 8] ==  9);
  BOOST_CHECK(res[ 9] == 12);
  BOOST_CHECK(res[10] == 13);
  BOOST_CHECK(res[11] == 14);
}


BOOST_AUTO_TEST_CASE(upper_mat_sequence_4)
{
  std::vector<std::size_t> res;
  auto test = [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      (res.push_back(Is), ...);
    };
  /*
   * |  0,  1,  2 |
   * |  3,  4,  5 |
   * |  6,  7,  8 |
   * |  9, 10, 11 |
   * | 12, 13, 14 |
   */
  test(mfmat::make_upper_mat_index_sequence<5, 3>());
  BOOST_CHECK(res.size() == 6);
  BOOST_CHECK(res[0] == 0);
  BOOST_CHECK(res[1] == 1);
  BOOST_CHECK(res[2] == 2);
  BOOST_CHECK(res[3] == 4);
  BOOST_CHECK(res[4] == 5);
  BOOST_CHECK(res[5] == 8);
}


BOOST_AUTO_TEST_CASE(upper_mat_no_diag_sequence_1)
{
  std::vector<std::size_t> res;
  auto test = [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      (res.push_back(Is), ...);
    };
  test(mfmat::make_upper_no_diag_mat_index_sequence<1>());
  BOOST_CHECK(res.empty());
}


BOOST_AUTO_TEST_CASE(upper_mat_no_diag_sequence_2)
{
  std::vector<std::size_t> res;
  auto test = [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      (res.push_back(Is), ...);
    };
  /*
   * | 0, 1, 2 |
   * | 3, 4, 5 |
   * | 6, 7, 8 |
   */
  test(mfmat::make_upper_no_diag_mat_index_sequence<3>());
  BOOST_CHECK(res.size() == 3);
  BOOST_CHECK(res[0] == 1);
  BOOST_CHECK(res[1] == 2);
  BOOST_CHECK(res[2] == 5);
}


BOOST_AUTO_TEST_CASE(upper_mat_no_diag_sequence_3)
{
  std::vector<std::size_t> res;
  auto test = [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      (res.push_back(Is), ...);
    };
  /*
   * |  0,  1,  2,  3,  4 |
   * |  5,  6,  7,  8,  9 |
   * | 10, 11, 12, 13, 14 |
   */
  test(mfmat::make_upper_no_diag_mat_index_sequence<3, 5>());
  BOOST_CHECK(res.size() == 9);
  BOOST_CHECK(res[0] ==  1);
  BOOST_CHECK(res[1] ==  2);
  BOOST_CHECK(res[2] ==  3);
  BOOST_CHECK(res[3] ==  4);
  BOOST_CHECK(res[4] ==  7);
  BOOST_CHECK(res[5] ==  8);
  BOOST_CHECK(res[6] ==  9);
  BOOST_CHECK(res[7] == 13);
  BOOST_CHECK(res[8] == 14);
}


BOOST_AUTO_TEST_CASE(upper_mat_no_diag_sequence_4)
{
  std::vector<std::size_t> res;
  auto test = [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      (res.push_back(Is), ...);
    };
  /*
   * |  0,  1,  2 |
   * |  3,  4,  5 |
   * |  6,  7,  8 |
   * |  9, 10, 11 |
   * | 12, 13, 14 |
   */
  test(mfmat::make_upper_no_diag_mat_index_sequence<5, 3>());
  BOOST_CHECK(res.size() == 3);
  BOOST_CHECK(res[0] == 1);
  BOOST_CHECK(res[1] == 2);
  BOOST_CHECK(res[2] == 5);
}


BOOST_AUTO_TEST_CASE(exclude_row_0)
{
  std::vector<std::size_t> res;
  auto test = [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      (res.push_back(Is), ...);
    };
  test(mfmat::remove_seq<owr, 3, 3, 0>(std::index_sequence<>{}));
  BOOST_CHECK(res.empty());
}


BOOST_AUTO_TEST_CASE(exclude_row_1)
{
  std::vector<std::size_t> res;
  auto test = [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      (res.push_back(Is), ...);
    };
  /*
   * | 0, 1, 2 |
   * | 3, 4, 5 |
   * | 6, 7, 8 |
   */
  auto base_seq = std::make_index_sequence<3 * 3>{};
  test(mfmat::remove_seq<owr, 3, 3, 0>(base_seq));
  BOOST_CHECK(res.size() == 6);
  BOOST_CHECK(res[0] == 3);
  BOOST_CHECK(res[1] == 4);
  BOOST_CHECK(res[2] == 5);
  BOOST_CHECK(res[3] == 6);
  BOOST_CHECK(res[4] == 7);
  BOOST_CHECK(res[5] == 8);
  res.clear();
  test(mfmat::remove_seq<owr, 3, 3, 1>(base_seq));
  BOOST_CHECK(res.size() == 6);
  BOOST_CHECK(res[0] == 0);
  BOOST_CHECK(res[1] == 1);
  BOOST_CHECK(res[2] == 2);
  BOOST_CHECK(res[3] == 6);
  BOOST_CHECK(res[4] == 7);
  BOOST_CHECK(res[5] == 8);
  res.clear();
  test(mfmat::remove_seq<owr, 3, 3, 2>(base_seq));
  BOOST_CHECK(res.size() == 6);
  BOOST_CHECK(res[0] == 0);
  BOOST_CHECK(res[1] == 1);
  BOOST_CHECK(res[2] == 2);
  BOOST_CHECK(res[3] == 3);
  BOOST_CHECK(res[4] == 4);
  BOOST_CHECK(res[5] == 5);
  res.clear();
}


BOOST_AUTO_TEST_CASE(exclude_row_2)
{
  std::vector<std::size_t> res;
  auto test = [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      (res.push_back(Is), ...);
    };
  /*
   * | 0, 1, 2 |
   * | 3, 4, 5 |
   * | 6, 7, 8 |
   */
  auto base_seq = std::make_index_sequence<3 * 3>{};
  auto seq1 = mfmat::remove_seq<owr, 3, 3, 0>(base_seq);
  auto seq2 = mfmat::remove_seq<owr, 3, 3, 2>(seq1);
  auto seq3 = mfmat::remove_seq<owr, 3, 3, 2>(seq2);
  test(seq3);
  BOOST_CHECK(res.size() == 3);
  BOOST_CHECK(res[0] == 3);
  BOOST_CHECK(res[1] == 4);
  BOOST_CHECK(res[2] == 5);
  res.clear();
  auto seq4 = mfmat::remove_seq<owr, 3, 3, 0>(base_seq);
  auto seq5 = mfmat::remove_seq<owr, 3, 3, 2>(seq4);
  auto seq6 = mfmat::remove_seq<owr, 3, 3, 1>(seq5);
  test(seq6);
  BOOST_CHECK(res.empty());
}


BOOST_AUTO_TEST_CASE(exclude_row_3)
{
  std::vector<std::size_t> res;
  auto test = [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      (res.push_back(Is), ...);
    };
  /*
   * | 0, 1 |
   * | 2, 3 |
   * | 4, 5 |
   * | 6, 7 |
   */
  auto seq1 = mfmat::remove_seq<owr, 4, 2, 1>(std::make_index_sequence<4 * 2>{});
  test(seq1);
  BOOST_CHECK(res.size() == 6);
  BOOST_CHECK(res[0] == 0);
  BOOST_CHECK(res[1] == 1);
  BOOST_CHECK(res[2] == 4);
  BOOST_CHECK(res[3] == 5);
  BOOST_CHECK(res[4] == 6);
  BOOST_CHECK(res[5] == 7);
  res.clear();
  auto seq2 = mfmat::remove_seq<owr, 4, 2, 2>(seq1);
  test(seq2);
  BOOST_CHECK(res.size() == 4);
  BOOST_CHECK(res[0] == 0);
  BOOST_CHECK(res[1] == 1);
  BOOST_CHECK(res[2] == 6);
  BOOST_CHECK(res[3] == 7);
  res.clear();
  auto seq3 = mfmat::remove_seq<owr, 4, 2, 0>(seq2);
  test(seq3);
  BOOST_CHECK(res.size() == 2);
  BOOST_CHECK(res[0] == 6);
  BOOST_CHECK(res[1] == 7);
  res.clear();
  auto seq4 = mfmat::remove_seq<owr, 4, 2, 3>(seq3);
  test(seq4);
  BOOST_CHECK(res.empty());
}


BOOST_AUTO_TEST_CASE(exclude_row_4)
{
  std::vector<std::size_t> res;
  auto test = [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      (res.push_back(Is), ...);
    };
  /*
   * | 0, 1, 2, 3 |
   * | 4, 5, 6, 7 |
   */
  auto base_seq = std::make_index_sequence<2 * 4>{};
  test(mfmat::remove_seq<owr, 2, 4, 0>(base_seq));
  BOOST_CHECK(res.size() == 4);
  BOOST_CHECK(res[0] == 4);
  BOOST_CHECK(res[1] == 5);
  BOOST_CHECK(res[2] == 6);
  BOOST_CHECK(res[3] == 7);
  res.clear();
  test(mfmat::remove_seq<owr, 2, 4, 1>(base_seq));
  BOOST_CHECK(res.size() == 4);
  BOOST_CHECK(res[0] == 0);
  BOOST_CHECK(res[1] == 1);
  BOOST_CHECK(res[2] == 2);
  BOOST_CHECK(res[3] == 3);
  res.clear();
  auto seq1 = mfmat::remove_seq<owr, 2, 4, 0>(base_seq);
  auto seq2 = mfmat::remove_seq<owr, 2, 4, 1>(seq1);
  test(seq2);
  BOOST_CHECK(res.empty());
}


BOOST_AUTO_TEST_CASE(exclude_column_0)
{
  std::vector<std::size_t> res;
  auto test = [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      (res.push_back(Is), ...);
    };
  test(mfmat::remove_seq<owc, 3, 3, 0>(std::index_sequence<>{}));
  BOOST_CHECK(res.empty());
}


BOOST_AUTO_TEST_CASE(exclude_column_1)
{
  std::vector<std::size_t> res;
  auto test = [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      (res.push_back(Is), ...);
    };
  /*
   * | 0, 1, 2 |
   * | 3, 4, 5 |
   * | 6, 7, 8 |
   */
  auto base_seq = std::make_index_sequence<3 * 3>{};
  test(mfmat::remove_seq<owc, 3, 3, 0>(base_seq));
  BOOST_CHECK(res.size() == 6);
  BOOST_CHECK(res[0] == 1);
  BOOST_CHECK(res[1] == 2);
  BOOST_CHECK(res[2] == 4);
  BOOST_CHECK(res[3] == 5);
  BOOST_CHECK(res[4] == 7);
  BOOST_CHECK(res[5] == 8);
  res.clear();
  test(mfmat::remove_seq<owc, 3, 3, 1>(base_seq));
  BOOST_CHECK(res.size() == 6);
  BOOST_CHECK(res[0] == 0);
  BOOST_CHECK(res[1] == 2);
  BOOST_CHECK(res[2] == 3);
  BOOST_CHECK(res[3] == 5);
  BOOST_CHECK(res[4] == 6);
  BOOST_CHECK(res[5] == 8);
  res.clear();
  test(mfmat::remove_seq<owc, 3, 3, 2>(base_seq));
  BOOST_CHECK(res.size() == 6);
  BOOST_CHECK(res[0] == 0);
  BOOST_CHECK(res[1] == 1);
  BOOST_CHECK(res[2] == 3);
  BOOST_CHECK(res[3] == 4);
  BOOST_CHECK(res[4] == 6);
  BOOST_CHECK(res[5] == 7);
  res.clear();
}


BOOST_AUTO_TEST_CASE(exclude_column_2)
{
  std::vector<std::size_t> res;
  auto test = [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      (res.push_back(Is), ...);
    };
  /*
   * | 0, 1, 2 |
   * | 3, 4, 5 |
   * | 6, 7, 8 |
   */
  auto base_seq = std::make_index_sequence<3 * 3>{};
  auto seq1 = mfmat::remove_seq<owc, 3, 3, 0>(base_seq);
  auto seq2 = mfmat::remove_seq<owc, 3, 3, 2>(seq1);
  auto seq3 = mfmat::remove_seq<owc, 3, 3, 2>(seq2);
  test(seq3);
  BOOST_CHECK(res.size() == 3);
  BOOST_CHECK(res[0] == 1);
  BOOST_CHECK(res[1] == 4);
  BOOST_CHECK(res[2] == 7);
  res.clear();
  auto seq4 = mfmat::remove_seq<owc, 3, 3, 0>(base_seq);
  auto seq5 = mfmat::remove_seq<owc, 3, 3, 2>(seq4);
  auto seq6 = mfmat::remove_seq<owc, 3, 3, 1>(seq5);
  test(seq6);
  BOOST_CHECK(res.empty());
}


BOOST_AUTO_TEST_CASE(exclude_colum_3)
{
  std::vector<std::size_t> res;
  auto test = [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      (res.push_back(Is), ...);
    };
  /*
   * | 0, 1, 2, 3 |
   * | 4, 5, 6, 7 |
   */
  auto seq1 = mfmat::remove_seq<owc, 2, 4, 1>(std::make_index_sequence<2 * 4>{});
  test(seq1);
  BOOST_CHECK(res.size() == 6);
  BOOST_CHECK(res[0] == 0);
  BOOST_CHECK(res[1] == 2);
  BOOST_CHECK(res[2] == 3);
  BOOST_CHECK(res[3] == 4);
  BOOST_CHECK(res[4] == 6);
  BOOST_CHECK(res[5] == 7);
  res.clear();
  auto seq2 = mfmat::remove_seq<owc, 2, 4, 2>(seq1);
  test(seq2);
  BOOST_CHECK(res.size() == 4);
  BOOST_CHECK(res[0] == 0);
  BOOST_CHECK(res[1] == 3);
  BOOST_CHECK(res[2] == 4);
  BOOST_CHECK(res[3] == 7);
  res.clear();
  auto seq3 = mfmat::remove_seq<owc, 2, 4, 0>(seq2);
  test(seq3);
  BOOST_CHECK(res.size() == 2);
  BOOST_CHECK(res[0] == 3);
  BOOST_CHECK(res[1] == 7);
  res.clear();
  auto seq4 = mfmat::remove_seq<owc, 2, 4, 3>(seq3);
  test(seq4);
  BOOST_CHECK(res.empty());
}

BOOST_AUTO_TEST_CASE(exclude_column_4)
{
  std::vector<std::size_t> res;
  auto test = [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      (res.push_back(Is), ...);
    };
  /*
   * | 0, 1 |
   * | 2, 3 |
   * | 4, 5 |
   * | 6, 7 |
   */
  auto base_seq = std::make_index_sequence<4 * 2>{};
  test(mfmat::remove_seq<owc, 4, 2, 0>(base_seq));
  BOOST_CHECK(res.size() == 4);
  BOOST_CHECK(res[0] == 1);
  BOOST_CHECK(res[1] == 3);
  BOOST_CHECK(res[2] == 5);
  BOOST_CHECK(res[3] == 7);
  res.clear();
  test(mfmat::remove_seq<owc, 4, 2, 1>(base_seq));
  BOOST_CHECK(res.size() == 4);
  BOOST_CHECK(res[0] == 0);
  BOOST_CHECK(res[1] == 2);
  BOOST_CHECK(res[2] == 4);
  BOOST_CHECK(res[3] == 6);
  res.clear();
  auto seq1 = mfmat::remove_seq<owc, 4, 2, 0>(base_seq);
  auto seq2 = mfmat::remove_seq<owc, 4, 2, 1>(seq1);
  test(seq2);
  BOOST_CHECK(res.empty());
}


BOOST_AUTO_TEST_CASE(exclude_mix)
{
  std::vector<std::size_t> res;
  auto test = [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
      (res.push_back(Is), ...);
    };
  /*
   * |  0,  1,  2 |
   * |  3,  4,  5 |
   * |  6,  7,  8 |
   * |  9, 10, 11 |
   * | 12, 13, 14 |
   */
  auto base_seq = std::make_index_sequence<5 * 3>{};
  auto seq1 = mfmat::remove_seq<owc, 5, 3, 1>(base_seq);
  auto seq2 = mfmat::remove_seq<owr, 5, 3, 2>(seq1);
  test(seq2);
  BOOST_CHECK(res.size() == 8);
  BOOST_CHECK(res[0] ==  0);
  BOOST_CHECK(res[1] ==  2);
  BOOST_CHECK(res[2] ==  3);
  BOOST_CHECK(res[3] ==  5);
  BOOST_CHECK(res[4] ==  9);
  BOOST_CHECK(res[5] == 11);
  BOOST_CHECK(res[6] == 12);
  BOOST_CHECK(res[7] == 14);
  res.clear();
  auto seq3 = mfmat::remove_seq<owr, 5, 3, 1>(seq2);
  auto seq4 = mfmat::remove_seq<owr, 5, 3, 3>(seq3);
  test(seq4);
  BOOST_CHECK(res.size() == 4);
  BOOST_CHECK(res[0] ==  0);
  BOOST_CHECK(res[1] ==  2);
  BOOST_CHECK(res[2] == 12);
  BOOST_CHECK(res[3] == 14);
  res.clear();
  auto seq5 = mfmat::remove_seq<owr, 5, 3, 0>(seq4);
  test(seq5);
  BOOST_CHECK(res.size() == 2);
  BOOST_CHECK(res[0] == 12);
  BOOST_CHECK(res[1] == 14);
  res.clear();
  auto seq6 = mfmat::remove_seq<owc, 5, 3, 0>(seq5);
  test(seq6);
  BOOST_CHECK(res.size() == 1);
  BOOST_CHECK(res[0] == 14);
  res.clear();
  auto seq7 = mfmat::remove_seq<owc, 5, 3, 2>(seq6);
  test(seq7);
  BOOST_CHECK(res.empty());
  res.clear();
  auto seq8 = mfmat::remove_seq<owr, 5, 3, 4>(seq6);
  test(seq8);
  BOOST_CHECK(res.empty());
}

BOOST_AUTO_TEST_SUITE_END()
