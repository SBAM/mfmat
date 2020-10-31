#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/tools/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(matrices_dot_product_test_suite)

constexpr auto owr = mfmat::op_way::row;
constexpr auto owc = mfmat::op_way::col;

BOOST_AUTO_TEST_CASE(simple_row_col)
{
  auto mat1 = mfmat::ct_mat<std::int32_t, 1, 3>
    ({
      { 1, 2, 3 },
     });
  auto mat2 = mfmat::ct_mat<std::int32_t, 3, 1>
    ({
      { 3 },
      { 2 },
      { 1 },
     });
  auto res = mfmat::dot<owr, 0, owc, 0>(mat1, mat2);
  BOOST_CHECK(res == 10);
}


BOOST_AUTO_TEST_CASE(simple_row_row)
{
  auto mat1 = mfmat::ct_mat<std::int32_t, 1, 3>
    ({
      { 1, 2, 3 },
     });
  auto mat2 = mfmat::ct_mat<std::int32_t, 1, 3>
    ({
      { 3, 2, 1 },
     });
  auto res = mfmat::dot<owr, 0, owr, 0>(mat1, mat2);
  BOOST_CHECK(res == 10);
}


BOOST_AUTO_TEST_CASE(simple_col_row)
{
  auto mat1 = mfmat::ct_mat<std::int32_t, 3, 1>
    ({
      { 1 },
      { 2 },
      { 3 },
     });
  auto mat2 = mfmat::ct_mat<std::int32_t, 1, 3>
    ({
      { 3, 2, 1 },
     });
  auto res = mfmat::dot<owc, 0, owr, 0>(mat1, mat2);
  BOOST_CHECK(res == 10);
}


BOOST_AUTO_TEST_CASE(simple_col_col)
{
  auto mat1 = mfmat::ct_mat<std::int32_t, 3, 1>
    ({
      { 1 },
      { 2 },
      { 3 },
     });
  auto mat2 = mfmat::ct_mat<std::int32_t, 3, 1>
    ({
      { 3 },
      { 2 },
      { 1 },
     });
  auto res = mfmat::dot<owc, 0, owc, 0>(mat1, mat2);
  BOOST_CHECK(res == 10);
}


BOOST_AUTO_TEST_CASE(multiple_row_col)
{
  auto mat1 = mfmat::ct_mat<std::int32_t, 6, 3>
    ({
      { -2, -1,  0 },
      { -1,  0,  1 },
      {  0,  1,  2 },
      {  0, -1, -2 },
      {  1,  0, -1 },
      {  2,  1,  0 }
     });
  auto mat2 = mfmat::ct_mat<std::int32_t, 3, 2>
    ({
      { 1, -1 },
      { 1, -1 },
      { 1, -1 }
     });
  std::int32_t res{};
  res = mfmat::dot<owr, 0, owc, 0>(mat1, mat2); BOOST_CHECK(res == -3);
  res = mfmat::dot<owr, 1, owc, 0>(mat1, mat2); BOOST_CHECK(res ==  0);
  res = mfmat::dot<owr, 2, owc, 0>(mat1, mat2); BOOST_CHECK(res ==  3);
  res = mfmat::dot<owr, 3, owc, 0>(mat1, mat2); BOOST_CHECK(res == -3);
  res = mfmat::dot<owr, 4, owc, 0>(mat1, mat2); BOOST_CHECK(res ==  0);
  res = mfmat::dot<owr, 5, owc, 0>(mat1, mat2); BOOST_CHECK(res ==  3);
  res = mfmat::dot<owr, 0, owc, 1>(mat1, mat2); BOOST_CHECK(res ==  3);
  res = mfmat::dot<owr, 1, owc, 1>(mat1, mat2); BOOST_CHECK(res ==  0);
  res = mfmat::dot<owr, 2, owc, 1>(mat1, mat2); BOOST_CHECK(res == -3);
  res = mfmat::dot<owr, 3, owc, 1>(mat1, mat2); BOOST_CHECK(res ==  3);
  res = mfmat::dot<owr, 4, owc, 1>(mat1, mat2); BOOST_CHECK(res ==  0);
  res = mfmat::dot<owr, 5, owc, 1>(mat1, mat2); BOOST_CHECK(res == -3);
}


BOOST_AUTO_TEST_CASE(simple_row_row_double)
{
  auto mat1 = mfmat::ct_mat<double, 1, 3>
    ({
      { 1.111, 2.222, 3.333 },
     });
  auto mat2 = mfmat::ct_mat<double, 1, 3>
    ({
      { 3.333, 2.222, 1.111 },
     });
  auto res = mfmat::dot<owr, 0, owr, 0>(mat1, mat2);
  BOOST_CHECK_CLOSE(res, 12.34321, 0.000001);
}

BOOST_AUTO_TEST_SUITE_END()
