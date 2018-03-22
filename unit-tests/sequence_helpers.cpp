#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mfmat

#include <iostream>
#include <vector>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/ct_sequence_helpers.hpp>

BOOST_AUTO_TEST_SUITE(sequence_helpers_test_suite)

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

BOOST_AUTO_TEST_SUITE_END()
