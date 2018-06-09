#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(construction_test_suite)

constexpr auto owr = mfmat::op_way::row;
constexpr auto owc = mfmat::op_way::col;

BOOST_AUTO_TEST_CASE(default_constructor)
{
  auto mat = mfmat::ct_mat<std::int32_t, 5, 10>();
  for (std::size_t i = 0; i < mat.row_count; ++i)
    for (std::size_t j = 0; j < mat.col_count; ++j)
    {
      auto cell = mat[{i, j}];
      BOOST_CHECK(cell == 0);
    }
}


BOOST_AUTO_TEST_CASE(identity_constructor)
{
  auto mat = mfmat::ct_mat<std::int64_t, 8, 8>(mfmat::identity_tag());
  for (std::size_t i = 0; i < mat.row_count; ++i)
    for (std::size_t j = 0; j < mat.col_count; ++j)
    {
      auto cell = mat[{i, j}];
      if (i == j)
        BOOST_CHECK(cell == 1);
      else
        BOOST_CHECK(cell == 0);
    }
}


BOOST_AUTO_TEST_CASE(identity_constructor_float)
{
  auto mat = mfmat::ct_mat<float, 8, 8>(mfmat::identity_tag());
  for (std::size_t i = 0; i < mat.row_count; ++i)
    for (std::size_t j = 0; j < mat.col_count; ++j)
    {
      auto cell = mat[{i, j}];
      if (i == j)
        BOOST_CHECK_CLOSE(cell, 1.0, 0.0000001);
      else
        BOOST_CHECK_CLOSE(cell, 0.0, 0.0000001);
    }
}


BOOST_AUTO_TEST_CASE(matrix_initializer_list_constructor)
{
  auto mat = mfmat::ct_mat<std::int16_t, 3, 4>
    ({
      { 0, 1, 2, 3 },
      { 4, 5, 6, 7 },
      { 8, 9, 0, 1 }
     });
  decltype(mat)::cell_t current = 0;
  for (std::size_t i = 0; i < mat.row_count; ++i)
    for (std::size_t j = 0; j < mat.col_count; ++j)
    {
      auto cell = mat[{i, j}];
      BOOST_CHECK(current++ % 10 == cell);
    }
}


BOOST_AUTO_TEST_CASE(copy_constructor)
{
  auto mat = mfmat::ct_mat<std::int16_t, 3, 4>
    ({
      { 0, 1, 2, 3 },
      { 4, 5, 6, 7 },
      { 8, 9, 0, 1 }
     });
  auto mat2(mat);
  decltype(mat2)::cell_t current = 0;
  for (std::size_t i = 0; i < mat2.row_count; ++i)
    for (std::size_t j = 0; j < mat2.col_count; ++j)
    {
      auto cell = mat2[{i, j}];
      BOOST_CHECK(current++ % 10 == cell);
    }
}


BOOST_AUTO_TEST_CASE(equal_op)
{
  auto mat = mfmat::ct_mat<std::int16_t, 3, 4>
    ({
      { 0, 1, 2, 3 },
      { 4, 5, 6, 7 },
      { 8, 9, 0, 1 }
     });
  decltype(mat) mat2;
  mat2 = mat;
  decltype(mat2)::cell_t current = 0;
  for (std::size_t i = 0; i < mat2.row_count; ++i)
    for (std::size_t j = 0; j < mat2.col_count; ++j)
    {
      auto cell = mat2[{i, j}];
      BOOST_CHECK(current++ % 10 == cell);
    }
}


BOOST_AUTO_TEST_CASE(sequence_constructor_1)
{
  auto mat = mfmat::ct_mat<std::int16_t, 3, 4>
    ({
      { 0, 1, 2, 3 },
      { 4, 5, 6, 7 },
      { 8, 9, 0, 1 }
     });
  auto base_seq = std::make_index_sequence<mat.row_count * mat.col_count>{};
  auto res = decltype(mat)(mat, base_seq);
  BOOST_CHECK(mat == res);
}


BOOST_AUTO_TEST_CASE(sequence_constructor_2)
{
  auto mat = mfmat::ct_mat<std::int32_t, 3, 4>
    ({
      { 0, 1, 2, 3 },
      { 4, 5, 6, 7 },
      { 8, 9, 0, 1 }
     });
  auto base_seq = std::make_index_sequence<mat.row_count * mat.col_count>{};
  auto seq = mfmat::remove_seq<owr, 3, 4, 1>(base_seq);
  auto expected = mfmat::ct_mat<std::int32_t, 2, 4>
    ({
      { 0, 1, 2, 3 },
      { 8, 9, 0, 1 }
     });
  auto res = mfmat::ct_mat<std::int32_t, 2, 4>(mat, seq);
  BOOST_CHECK(expected == res);
}


BOOST_AUTO_TEST_CASE(sequence_constructor_3)
{
  auto mat = mfmat::ct_mat<std::int64_t, 3, 4>
    ({
      { 0, 1, 2, 3 },
      { 4, 5, 6, 7 },
      { 8, 9, 0, 1 }
     });
  auto base_seq = std::make_index_sequence<mat.row_count * mat.col_count>{};
  auto seq1 = mfmat::remove_seq<owc, 3, 4, 1>(base_seq);
  auto seq2 = mfmat::remove_seq<owc, 3, 4, 2>(seq1);
  auto expected = mfmat::ct_mat<std::int64_t, 3, 2>
    ({
      { 0, 3 },
      { 4, 7 },
      { 8, 1 }
     });
  auto res = mfmat::ct_mat<std::int64_t, 3, 2>(mat, seq2);
  BOOST_CHECK(expected == res);
}


BOOST_AUTO_TEST_CASE(sequence_constructor_4)
{
  auto mat = mfmat::ct_mat<float, 3, 4>
    ({
      { 0, 1, 2, 3 },
      { 4, 5, 6, 7 },
      { 8, 9, 0, 1 }
     });
  auto base_seq = std::make_index_sequence<mat.row_count * mat.col_count>{};
  auto seq1 = mfmat::remove_seq<owc, 3, 4, 1>(base_seq);
  auto seq2 = mfmat::remove_seq<owc, 3, 4, 2>(seq1);
  auto seq3 = mfmat::remove_seq<owr, 3, 4, 0>(seq2);
  auto seq4 = mfmat::remove_seq<owr, 3, 4, 1>(seq3);
  auto expected = mfmat::ct_mat<float, 1, 2>
    ({
      { 8, 1 }
     });
  auto res = mfmat::ct_mat<float, 1, 2>(mat, seq4);
  BOOST_CHECK(expected == res);
}


BOOST_AUTO_TEST_CASE(sequence_constructor_5)
{
  auto mat = mfmat::ct_mat<double, 2, 2>
    ({
      {  8, 16 },
      { 32, 64 }
     });
  auto base_seq = std::index_sequence<0, 1, 0, 1,
                                      2, 3, 2, 3,
                                      0, 1, 0, 1,
                                      2, 3, 2, 3>{};
  auto expected = mfmat::ct_mat<double, 4, 4>
    ({
      {  8, 16,  8, 16 },
      { 32, 64, 32, 64 },
      {  8, 16,  8, 16 },
      { 32, 64, 32, 64 }
     });
  auto res = mfmat::ct_mat<double, 4, 4>(mat, base_seq);
  BOOST_CHECK(expected == res);
}

BOOST_AUTO_TEST_SUITE_END()
