#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(matrices_multiply_suite)

BOOST_AUTO_TEST_CASE(multiply_simple)
{
  auto mat = mfmat::ct_mat<std::int32_t, 2, 3>
    ({
      { 1, 2, 3 },
      { 4, 5, 6 }
     });
  auto mat2 = mfmat::ct_mat<std::int32_t, 3, 2>
    ({
      { 7, 8 },
      { 9, 10 },
      { 11, 12 }
     });
  auto mat3 = mfmat::ct_mat<std::int32_t, 2, 2>
    ({
      { 58, 64 },
      { 139, 154 }
     });
  auto res = mat * mat2;
  BOOST_CHECK(res == mat3);
}


BOOST_AUTO_TEST_CASE(multiply_type_promotion)
{
  auto mat = mfmat::ct_mat<std::int8_t, 1, 2>
    ({
      { 100, 100 }
     });
  auto mat2 = mfmat::ct_mat<std::int32_t, 2, 1>
    ({
      { 100 },
      { 100 }
     });
  auto mat3 = mfmat::ct_mat<std::int32_t, 1, 1>({ { 20000 } });
  auto res = mat * mat2;
  BOOST_CHECK(res == mat3);
  auto mat4 = mfmat::ct_mat<std::int32_t, 1, 2>
    ({
      { 100, 100 }
     });
  auto mat5 = mfmat::ct_mat<std::int8_t, 2, 1>
    ({
      { 100 },
      { 100 }
     });
  auto res2 = mat4 * mat5;
  BOOST_CHECK(res2 == mat3);
}


BOOST_AUTO_TEST_CASE(multiply_double)
{
  auto mat = mfmat::ct_mat<double, 3, 4>
    ({
      {  1.2, -2.4,  2.4, -1.8 },
      {  4.2, -3.1,  0.9, -0.8 },
      {  1.1, -1.2,  2.2, -3.3 }
     });
  auto mat2 = mfmat::ct_mat<double, 4, 4>
    ({
      {  1.0,  2.1, -0.8, -0.6 },
      {  0.5,  0.3,  1.5, -2.5 },
      {  0.1, -0.1,  2.5,  1.3 },
      {  2.3, -0.7, -0.5,  0.2 }
     });
  auto mat3 = mfmat::ct_mat<double, 3, 4>
    ({
      { -3.90,  2.82,  2.34,  8.04 },
      {  0.90,  8.36, -5.36,  6.24 },
      { -6.87,  4.04,  4.47,  4.54 }
     });
  auto res = mat * mat2;
  BOOST_CHECK(res == mat3);
}


BOOST_AUTO_TEST_CASE(multiply_outer)
{
  auto mat = mfmat::ct_mat<std::int32_t, 3, 2>
    ({
      {  1, 2 },
      {  3, 4 },
      {  5, 6 }
     });
  auto mat2 = mfmat::ct_mat<std::int32_t, 2, 3>
    ({
      {  1, 3, 5 },
      {  2, 4, 6 }
     });
  auto mat3 = mfmat::ct_mat<std::int32_t, 3, 3>
    ({
      {  5, 11, 17 },
      { 11, 25, 39 },
      { 17, 39, 61 }
     });
  auto res = mat * mat2;
  BOOST_CHECK(res == mat3);
}

BOOST_AUTO_TEST_SUITE_END()
