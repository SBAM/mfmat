#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(construction_test_suite)

BOOST_AUTO_TEST_CASE(default_constructor)
{
  auto mat = mfmat::dense_matrix<std::int32_t, 5, 10>();
  for (std::size_t i = 0; i < mat.row_count; ++i)
    for (std::size_t j = 0; j < mat.col_count; ++j)
    {
      auto cell = mat[{i, j}];
      BOOST_CHECK(cell == 0);
    }
}


BOOST_AUTO_TEST_CASE(identity_constructor)
{
  auto mat = mfmat::dense_matrix<std::int64_t, 8, 8>(mfmat::identity_tag());
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


BOOST_AUTO_TEST_CASE(matrix_initializer_list_constructor)
{
  auto mat = mfmat::dense_matrix<std::int16_t, 3, 4>
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


BOOST_AUTO_TEST_CASE(vector_initializer_list_constructor)
{
  auto mat = mfmat::dense_matrix<std::int16_t, 7, 1>({ 0, 1, 2, 3, 4, 5, 6 });
  decltype(mat)::cell_t current = 0;
  for (std::size_t i = 0; i < mat.row_count; ++i)
  {
    auto cell = mat[{i, 0}];
    BOOST_CHECK(current++ == cell);
  }
}


BOOST_AUTO_TEST_CASE(copy_constructor)
{
  auto mat = mfmat::dense_matrix<std::int16_t, 3, 4>
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
  auto mat = mfmat::dense_matrix<std::int16_t, 3, 4>
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

BOOST_AUTO_TEST_SUITE_END()
