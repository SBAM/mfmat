#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/tools/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

namespace but = boost::unit_test;

BOOST_AUTO_TEST_SUITE(cl_scalar_ops_test_suite)

namespace
{

  void setup()
  {
    mfmat::cl_default_gpu_setter::instance();
    mfmat::cl_kernels_store::instance();
  }

} // !namespace anonymous

BOOST_AUTO_TEST_CASE(add_and_store, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{32, 32};
  mat_f += 2.2f;
  for (auto i : mat_f)
    BOOST_CHECK(mfmat::are_equal(i, 2.2f));
  auto mat_d = mfmat::cl_mat_d{64, 64};
  mat_d += 3.3;
  for (auto i : mat_d)
    BOOST_CHECK(mfmat::are_equal(i, 3.3));
}


BOOST_AUTO_TEST_CASE(sub_and_store, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{64, 64};
  mat_f -= 1.1f;
  for (auto i : mat_f)
    BOOST_CHECK(mfmat::are_equal(i, -1.1f));
  auto mat_d = mfmat::cl_mat_d{32, 32};
  mat_d -= -4.4;
  for (auto i : mat_d)
    BOOST_CHECK(mfmat::are_equal(i, 4.4));
}


BOOST_AUTO_TEST_CASE(mul_and_store, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{64, 32};
  mat_f += 1.0f;
  mat_f *= 5.5f;
  for (auto i : mat_f)
    BOOST_CHECK(mfmat::are_equal(i, 5.5f));
  auto mat_d = mfmat::cl_mat_d{32, 64};
  mat_d -= 1.0;
  mat_d *= -6.6;
  for (auto i : mat_d)
    BOOST_CHECK(mfmat::are_equal(i, 6.6));
}


BOOST_AUTO_TEST_CASE(div_and_store, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{33, 65};
  mat_f += 1.0f;
  mat_f /= 2.0f;
  for (auto i : mat_f)
    BOOST_CHECK(mfmat::are_equal(i, 0.5f));
  auto mat_d = mfmat::cl_mat_d{65, 33};
  mat_d -= 1.0;
  mat_d /= -2.0;
  for (auto i : mat_d)
    BOOST_CHECK(mfmat::are_equal(i, 0.5));
}

BOOST_AUTO_TEST_SUITE_END()
