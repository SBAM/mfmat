#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(matrices_dot_product_test_suite)

BOOST_AUTO_TEST_CASE(simple_row_col)
{
  auto mat1 = mfmat::dense_matrix<std::int32_t, 1, 3>
    ({
      { 1, 2, 3 },
     });
  auto mat2 = mfmat::dense_matrix<std::int32_t, 3, 1>
    ({
      { 3 },
      { 2 },
      { 1 },
     });
  auto res = mfmat::dot<mfmat::dot_spec::row_col, 0, 0>(mat1, mat2);
  BOOST_CHECK(res == 10);
}


BOOST_AUTO_TEST_CASE(simple_row_row)
{
  auto mat1 = mfmat::dense_matrix<std::int32_t, 1, 3>
    ({
      { 1, 2, 3 },
     });
  auto mat2 = mfmat::dense_matrix<std::int32_t, 1, 3>
    ({
      { 3, 2, 1 },
     });
  auto res = mfmat::dot<mfmat::dot_spec::row_row, 0, 0>(mat1, mat2);
  BOOST_CHECK(res == 10);
}


BOOST_AUTO_TEST_CASE(simple_col_row)
{
  auto mat1 = mfmat::dense_matrix<std::int32_t, 3, 1>
    ({
      { 1 },
      { 2 },
      { 3 },
     });
  auto mat2 = mfmat::dense_matrix<std::int32_t, 1, 3>
    ({
      { 3, 2, 1 },
     });
  auto res = mfmat::dot<mfmat::dot_spec::col_row, 0, 0>(mat1, mat2);
  BOOST_CHECK(res == 10);
}


BOOST_AUTO_TEST_CASE(simple_col_col)
{
  auto mat1 = mfmat::dense_matrix<std::int32_t, 3, 1>
    ({
      { 1 },
      { 2 },
      { 3 },
     });
  auto mat2 = mfmat::dense_matrix<std::int32_t, 3, 1>
    ({
      { 3 },
      { 2 },
      { 1 },
     });
  auto res = mfmat::dot<mfmat::dot_spec::col_col, 0, 0>(mat1, mat2);
  BOOST_CHECK(res == 10);
}


BOOST_AUTO_TEST_CASE(multiple_row_col)
{
  auto mat1 = mfmat::dense_matrix<std::int32_t, 6, 3>
    ({
      { -2, -1,  0 },
      { -1,  0,  1 },
      {  0,  1,  2 },
      {  0, -1, -2 },
      {  1,  0, -1 },
      {  2,  1,  0 }
     });
  auto mat2 = mfmat::dense_matrix<std::int32_t, 3, 2>
    ({
      { 1, -1 },
      { 1, -1 },
      { 1, -1 }
     });
  constexpr auto ds = mfmat::dot_spec::row_col;
  std::int32_t res{};
  res = mfmat::dot<ds, 0, 0>(mat1, mat2); BOOST_CHECK(res == -3);
  res = mfmat::dot<ds, 1, 0>(mat1, mat2); BOOST_CHECK(res ==  0);
  res = mfmat::dot<ds, 2, 0>(mat1, mat2); BOOST_CHECK(res ==  3);
  res = mfmat::dot<ds, 3, 0>(mat1, mat2); BOOST_CHECK(res == -3);
  res = mfmat::dot<ds, 4, 0>(mat1, mat2); BOOST_CHECK(res ==  0);
  res = mfmat::dot<ds, 5, 0>(mat1, mat2); BOOST_CHECK(res ==  3);
  res = mfmat::dot<ds, 0, 1>(mat1, mat2); BOOST_CHECK(res ==  3);
  res = mfmat::dot<ds, 1, 1>(mat1, mat2); BOOST_CHECK(res ==  0);
  res = mfmat::dot<ds, 2, 1>(mat1, mat2); BOOST_CHECK(res == -3);
  res = mfmat::dot<ds, 3, 1>(mat1, mat2); BOOST_CHECK(res ==  3);
  res = mfmat::dot<ds, 4, 1>(mat1, mat2); BOOST_CHECK(res ==  0);
  res = mfmat::dot<ds, 5, 1>(mat1, mat2); BOOST_CHECK(res == -3);
}


BOOST_AUTO_TEST_CASE(simple_row_row_double)
{
  auto mat1 = mfmat::dense_matrix<double, 1, 3>
    ({
      { 1.111, 2.222, 3.333 },
     });
  auto mat2 = mfmat::dense_matrix<double, 1, 3>
    ({
      { 3.333, 2.222, 1.111 },
     });
  auto res = mfmat::dot<mfmat::dot_spec::row_row, 0, 0>(mat1, mat2);
  BOOST_CHECK_CLOSE(res, 12.34321, 0.000001);
}

BOOST_AUTO_TEST_SUITE_END()
