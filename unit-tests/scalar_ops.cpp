#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(scalar_ops_test_suite)

BOOST_AUTO_TEST_CASE(add_and_store)
{
  auto mat = mfmat::ct_mat<std::int32_t, 2, 3>
    ({
      { 1, 1, 1 },
      { 1, 1, 1 }
     });
  mat += 1;
  for (std::size_t i = 0; i < mat.row_count; ++i)
    for (std::size_t j = 0; j < mat.col_count; ++j)
    {
      auto cell = mat[{i, j}];
      BOOST_CHECK(cell == 2);
    }
}


BOOST_AUTO_TEST_CASE(substract_and_store)
{
  auto mat = mfmat::ct_mat<std::int32_t, 2, 3>
    ({
      { -1, -1, -1 },
      { -1, -1, -1 }
     });
  mat -= 1;
  for (std::size_t i = 0; i < mat.row_count; ++i)
    for (std::size_t j = 0; j < mat.col_count; ++j)
    {
      auto cell = mat[{i, j}];
      BOOST_CHECK(cell == -2);
    }
}


BOOST_AUTO_TEST_CASE(multiply_and_store)
{
  auto mat = mfmat::ct_mat<std::int32_t, 2, 3>
    ({
      { 2, 2, 2 },
      { 2, 2, 2 }
     });
  mat *= 3;
  for (std::size_t i = 0; i < mat.row_count; ++i)
    for (std::size_t j = 0; j < mat.col_count; ++j)
    {
      auto cell = mat[{i, j}];
      BOOST_CHECK(cell == 6);
    }
}


BOOST_AUTO_TEST_CASE(divide_and_store)
{
  auto mat = mfmat::ct_mat<std::int32_t, 2, 3>
    ({
      { 8, 8, 8 },
      { 8, 8, 8 }
     });
  mat /= 2;
  for (std::size_t i = 0; i < mat.row_count; ++i)
    for (std::size_t j = 0; j < mat.col_count; ++j)
    {
      auto cell = mat[{i, j}];
      BOOST_CHECK(cell == 4);
    }
}


BOOST_AUTO_TEST_CASE(add)
{
  auto mat = mfmat::ct_mat<std::int32_t, 2, 3>
    ({
      { 1, 1, 1 },
      { 1, 1, 1 }
     });
  auto mat2 = mat + 1;
  auto mat3 = 1 + mat + 1;
  for (std::size_t i = 0; i < mat.row_count; ++i)
    for (std::size_t j = 0; j < mat.col_count; ++j)
    {
      auto cell2 = mat2[{i, j}];
      auto cell3 = mat3[{i, j}];
      BOOST_CHECK(cell2 == 2);
      BOOST_CHECK(cell3 == 3);
    }
}


BOOST_AUTO_TEST_CASE(add_double)
{
  auto mat = mfmat::ct_mat<double, 2, 3>
    ({
      { 1.1, 1.1, 1.1 },
      { 1.1, 1.1, 1.1 }
     });
  auto mat2 = mat + 1.1;
  auto mat3 = 1.1 + mat + 1.1;
  for (std::size_t i = 0; i < mat.row_count; ++i)
    for (std::size_t j = 0; j < mat.col_count; ++j)
    {
      auto cell2 = mat2[{i, j}];
      auto cell3 = mat3[{i, j}];
      BOOST_CHECK_CLOSE(cell2, 2.2, 0.00001);
      BOOST_CHECK_CLOSE(cell3, 3.3, 0.00001);
    }
}


BOOST_AUTO_TEST_CASE(substract)
{
  auto mat = mfmat::ct_mat<std::int32_t, 2, 3>
    ({
      { 5, 5, 5 },
      { 5, 5, 5 }
     });
  auto mat2 = mat - 2;
  auto mat3 = mat - 1 - 2;
  for (std::size_t i = 0; i < mat.row_count; ++i)
    for (std::size_t j = 0; j < mat.col_count; ++j)
    {
      auto cell2 = mat2[{i, j}];
      auto cell3 = mat3[{i, j}];
      BOOST_CHECK(cell2 == 3);
      BOOST_CHECK(cell3 == 2);
    }
}


BOOST_AUTO_TEST_CASE(substract_double)
{
  auto mat = mfmat::ct_mat<double, 2, 3>
    ({
      { 5.5, 5.5, 5.5 },
      { 5.5, 5.5, 5.5 }
     });
  auto mat2 = mat - 2.2;
  auto mat3 = mat - 1.1 - 2.2;
  for (std::size_t i = 0; i < mat.row_count; ++i)
    for (std::size_t j = 0; j < mat.col_count; ++j)
    {
      auto cell2 = mat2[{i, j}];
      auto cell3 = mat3[{i, j}];
      BOOST_CHECK_CLOSE(cell2, 3.3, 0.00001);
      BOOST_CHECK_CLOSE(cell3, 2.2, 0.00001);
    }
}


BOOST_AUTO_TEST_CASE(multiply)
{
  auto mat = mfmat::ct_mat<std::int32_t, 2, 3>
    ({
      { 2, 2, 2 },
      { 2, 2, 2 }
     });
  auto mat2 = mat * 2;
  auto mat3 = 2 * mat * 2;
  for (std::size_t i = 0; i < mat.row_count; ++i)
    for (std::size_t j = 0; j < mat.col_count; ++j)
    {
      auto cell2 = mat2[{i, j}];
      auto cell3 = mat3[{i, j}];
      BOOST_CHECK(cell2 == 4);
      BOOST_CHECK(cell3 == 8);
    }
}


BOOST_AUTO_TEST_CASE(multiply_double)
{
  auto mat = mfmat::ct_mat<double, 2, 3>
    ({
      { 2.2, 2.2, 2.2 },
      { 2.2, 2.2, 2.2 }
     });
  auto mat2 = mat * 2.2;
  auto mat3 = 2.2 * mat * 2.2;
  for (std::size_t i = 0; i < mat.row_count; ++i)
    for (std::size_t j = 0; j < mat.col_count; ++j)
    {
      auto cell2 = mat2[{i, j}];
      auto cell3 = mat3[{i, j}];
      BOOST_CHECK_CLOSE(cell2, 4.84, 0.00001);
      BOOST_CHECK_CLOSE(cell3, 10.648, 0.00001);
    }
}


BOOST_AUTO_TEST_CASE(divide)
{
  auto mat = mfmat::ct_mat<std::int32_t, 2, 3>
    ({
      { 8, 8, 8 },
      { 8, 8, 8 }
     });
  auto mat2 = mat / 2;
  auto mat3 = mat2 / 2;
  for (std::size_t i = 0; i < mat.row_count; ++i)
    for (std::size_t j = 0; j < mat.col_count; ++j)
    {
      auto cell2 = mat2[{i, j}];
      auto cell3 = mat3[{i, j}];
      BOOST_CHECK(cell2 == 4);
      BOOST_CHECK(cell3 == 2);
    }
}


BOOST_AUTO_TEST_CASE(divide_double)
{
  auto mat = mfmat::ct_mat<double, 2, 3>
    ({
      { 8.8, 8.8, 8.8 },
      { 8.8, 8.8, 8.8 }
     });
  auto mat2 = mat / 2.2;
  auto mat3 = mat2 / 2.0;
  for (std::size_t i = 0; i < mat.row_count; ++i)
    for (std::size_t j = 0; j < mat.col_count; ++j)
    {
      auto cell2 = mat2[{i, j}];
      auto cell3 = mat3[{i, j}];
      BOOST_CHECK_CLOSE(cell2, 4.0, 0.00001);
      BOOST_CHECK_CLOSE(cell3, 2.0, 0.00001);
    }
}

BOOST_AUTO_TEST_SUITE_END()
