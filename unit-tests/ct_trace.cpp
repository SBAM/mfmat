#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/tools/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(trace_test_suite)

BOOST_AUTO_TEST_CASE(trace_2x2)
{
  auto mat1 = mfmat::ct_mat<std::int32_t, 2, 2>(mfmat::identity_tag{});
  auto res1 = mat1.trace();
  BOOST_CHECK(res1 == 2);
  auto mat2 = mfmat::ct_mat<std::int32_t, 2, 2>
    ({
      { 1, 2 },
      { 3, 4 }
     });
  auto res2 = mat2.trace();
  BOOST_CHECK(res2 == 5);
  auto mat3 = mfmat::ct_mat<std::int32_t, 2, 2>();
  auto res3 = mat3.trace();
  BOOST_CHECK(res3 == 0);
}


BOOST_AUTO_TEST_CASE(trace_3x3)
{
  auto mat = mfmat::ct_mat<std::int32_t, 3, 3>
    ({
      { -2,  2, -3 },
      { -1,  1,  3 },
      {  2,  0, -1 }
     });
  auto res = mat.trace();
  BOOST_CHECK(res == -2);
}


BOOST_AUTO_TEST_CASE(trace_5x5)
{
  auto mat = mfmat::ct_mat<double, 5, 5>
    ({
      {  0.5,  0.6,  0.7,  0.8,  0.9 },
      { -0.1, -0.3, -0.2, -0.4, -0.5 },
      {  2.2, -3.3,  4.4, -5.5,  6.6 },
      {  0.0,  2.5,  0.0, -2.5,  0.0 },
      {  1.1,  1.2, -1.1, -1.2,  0.1 }
     });
  auto res = mat.trace();
  BOOST_CHECK_CLOSE(res, 2.2, 0.001);
}

BOOST_AUTO_TEST_SUITE_END()
