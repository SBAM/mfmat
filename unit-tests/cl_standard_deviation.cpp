#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/tools/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

namespace but = boost::unit_test;

BOOST_AUTO_TEST_SUITE(ct_standard_deviation_test_suite)

namespace
{

  void setup()
  {
    mfmat::cl_default_gpu_setter::instance();
    mfmat::cl_kernels_store::instance();
  }

} // !namespace anonymous

BOOST_AUTO_TEST_CASE(identity_1x1_standard_deviation, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{1, mfmat::identity_tag{}};
  mat_f *= 2.0f;
  auto sd_f = mfmat::std_dev(mat_f);
  BOOST_CHECK_EQUAL(mat_f.get_row_count(), 1);
  BOOST_CHECK_EQUAL(mat_f.get_col_count(), 1);
  BOOST_CHECK(mfmat::is_zero(sd_f.get(0, 0)));
  auto mat_d = mfmat::cl_mat_d{1, mfmat::identity_tag{}};
  mat_d *= 3.0;
  auto sd_d = mfmat::std_dev(mat_d);
  BOOST_CHECK_EQUAL(mat_d.get_row_count(), 1);
  BOOST_CHECK_EQUAL(mat_d.get_col_count(), 1);
  BOOST_CHECK(mfmat::is_zero(sd_d.get(0, 0)));
}


BOOST_AUTO_TEST_CASE(standard_deviation_3x3)
{
  auto mat_f = mfmat::cl_mat_f
    ({
      { 1.0f, 2.0f, 3.0f },
      { 4.0f, 5.0f, 6.0f },
      { 7.0f, 8.0f, 9.0f }
    });
  auto sd_f = mfmat::std_dev(mat_f);
  BOOST_CHECK_EQUAL(sd_f.get_row_count(), 1);
  BOOST_CHECK_EQUAL(sd_f.get_col_count(), 3);
  for (auto cell : sd_f)
    BOOST_CHECK_CLOSE(cell, 2.4495, 0.001);
}


BOOST_AUTO_TEST_CASE(standard_deviation_5x3)
{
  auto mat_d = mfmat::cl_mat_d
    ({
      { 90.0, 80.0, 40.0 },
      { 90.0, 60.0, 80.0 },
      { 60.0, 50.0, 70.0 },
      { 30.0, 40.0, 70.0 },
      { 30.0, 20.0, 90.0 }
     });
  auto sd_d = mfmat::std_dev(mat_d);
  BOOST_CHECK_EQUAL(sd_d.get_row_count(), 1);
  BOOST_CHECK_EQUAL(sd_d.get_col_count(), 3);
  BOOST_CHECK_CLOSE(sd_d.get(0, 0), 26.8328, 0.001);
  BOOST_CHECK_CLOSE(sd_d.get(0, 1), 20.0, 0.001);
  BOOST_CHECK_CLOSE(sd_d.get(0, 2), 16.7332, 0.001);
}


BOOST_AUTO_TEST_CASE(stddev_empty_matrix, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{0, 0};
  BOOST_CHECK_THROW(mfmat::std_dev(mat_f), std::runtime_error);
  auto mat_d = mfmat::cl_mat_d{0, 0};
  BOOST_CHECK_THROW(mfmat::std_dev(mat_d), std::runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
