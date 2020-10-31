#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/tools/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

namespace but = boost::unit_test;

BOOST_AUTO_TEST_SUITE(cl_compare_ops_test_suite)

namespace
{

  void setup()
  {
    mfmat::cl_default_gpu_setter::instance();
    mfmat::cl_kernels_store::instance();
  }

} // !namespace anonymous

BOOST_AUTO_TEST_CASE(float_dim_diff, * but::fixture(&setup))
{
  auto mat = mfmat::cl_mat_f(1);
  auto mat2 = mfmat::cl_mat_f(2);
  BOOST_CHECK(mat != mat2);
}


BOOST_AUTO_TEST_CASE(float_equal, * but::fixture(&setup))
{
  auto mat = mfmat::cl_mat_f(250, mfmat::identity_tag{});
  auto mat2 = mat;
  BOOST_CHECK(mat == mat2);
}


BOOST_AUTO_TEST_CASE(float_diff, * but::fixture(&setup))
{
  auto mat = mfmat::cl_mat_f(300, mfmat::identity_tag{});
  auto mat2 = mat;
  mat2.get(150, 150) += 0.00001f;
  BOOST_CHECK(mat != mat2);
}


BOOST_AUTO_TEST_CASE(double_equal, * but::fixture(&setup))
{
  auto mat = mfmat::cl_mat_d(350);
  mat += 1000;
  auto mat2 = mat;
  BOOST_CHECK(mat == mat2);
}


BOOST_AUTO_TEST_CASE(double_diff, * but::fixture(&setup))
{
  auto mat = mfmat::cl_mat_d(400);
  mat += 1000.0;
  auto mat2 = mat + 10000.0;
  BOOST_CHECK(mat != mat2);
}


BOOST_AUTO_TEST_CASE(double_diff_high, * but::fixture(&setup))
{
  auto mat = mfmat::cl_mat_d(450);
  mat += 999999999999999.0;
  auto mat2 = mfmat::cl_mat_d(450);
  mat2 += 999999999999998.0;
  BOOST_CHECK(mat != mat2);
}


BOOST_AUTO_TEST_CASE(double_diff_low, * but::fixture(&setup))
{
  auto mat = mfmat::cl_mat_d(500);
  mat += 1.0e-15;
  auto mat2 = mfmat::cl_mat_d(500);
  mat2 -= 1.0e-15;
  BOOST_CHECK(mat != mat2);
}

BOOST_AUTO_TEST_SUITE_END()
