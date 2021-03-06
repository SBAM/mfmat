#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/tools/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(ct_compare_ops_test_suite)

BOOST_AUTO_TEST_CASE(integer_compare)
{
  auto mat = mfmat::ct_mat<std::int32_t, 2, 3>
    ({
      { 1, 1, 1 },
      { 1, 1, 1 }
     });
  auto mat2 = mat;
  BOOST_CHECK(mat == mat2);
}


BOOST_AUTO_TEST_CASE(integer_diff)
{
  auto mat = mfmat::ct_mat<std::int32_t, 1, 1>({{ 0 }});
  auto mat2 = mfmat::ct_mat<std::int32_t, 1, 1>({{ 1 }});
  BOOST_CHECK(mat != mat2);
}


BOOST_AUTO_TEST_CASE(double_compare)
{
  auto mat = mfmat::ct_mat<double, 2, 4>
    ({
      {  1.0e15,  1.0,  1.0e-15,  0.0 },
      { -1.0e15, -1.0, -1.0e-15, -0.0 }
     });
  auto mat2 = mat;
  BOOST_CHECK(mat == mat2);
}


BOOST_AUTO_TEST_CASE(double_diff1)
{
  auto mat = mfmat::ct_mat<double, 1, 1>({{  999999999999999.0 }});
  auto mat2 = mfmat::ct_mat<double, 1, 1>({{ 999999999999998.0 }});
  BOOST_CHECK(mat != mat2);
}


BOOST_AUTO_TEST_CASE(double_diff2)
{
  auto mat = mfmat::ct_mat<double, 1, 1>({{  -999999999999999.0 }});
  auto mat2 = mfmat::ct_mat<double, 1, 1>({{ -999999999999998.0 }});
  BOOST_CHECK(mat != mat2);
}


BOOST_AUTO_TEST_CASE(double_diff3)
{
  auto mat = mfmat::ct_mat<double, 1, 1>({{  1.0e-15 }});
  auto mat2 = mfmat::ct_mat<double, 1, 1>({{ 2.0e-15 }});
  BOOST_CHECK(mat != mat2);
}


BOOST_AUTO_TEST_CASE(double_diff4)
{
  auto mat = mfmat::ct_mat<double, 1, 1>({{  -1.0e-15 }});
  auto mat2 = mfmat::ct_mat<double, 1, 1>({{ -2.0e-15 }});
  BOOST_CHECK(mat != mat2);
}

BOOST_AUTO_TEST_SUITE_END()
