#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

namespace but = boost::unit_test;

BOOST_AUTO_TEST_SUITE(cl_stddev_center_test_suite)

namespace
{

  void setup()
  {
    mfmat::cl_default_gpu_setter::instance();
    mfmat::cl_kernels_store::instance();
  }

} // !namespace anonymous

BOOST_AUTO_TEST_CASE(identity_2x2_stddev_center, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{2, mfmat::identity_tag{}};
  auto res_f = mat_f * 2.0f;
  res_f -= 1.0f;
  mat_f.stddev_center();
  auto zero_f = res_f - mat_f;
  for (auto cell : zero_f)
    BOOST_CHECK(mfmat::is_zero(cell));
  auto mat_d = mfmat::cl_mat_d{2, mfmat::identity_tag{}};
  auto res_d = mat_d * 2.0;
  res_d -= 1.0;
  mat_d.stddev_center();
  auto zero_d = res_d - mat_d;
  for (auto cell : zero_d)
    BOOST_CHECK(mfmat::is_zero(cell));
}


BOOST_AUTO_TEST_CASE(stddev_center_3x3)
{
  auto mat_f = mfmat::cl_mat_f
    ({
      { 1.0f, 2.0f, 3.0f },
      { 4.0f, 5.0f, 6.0f },
      { 7.0f, 8.0f, 9.0f }
     });
  mat_f.stddev_center();
  BOOST_CHECK_EQUAL(mat_f.get_row_count(), 3);
  BOOST_CHECK_EQUAL(mat_f.get_col_count(), 3);
  auto res_f = mfmat::cl_mat_f
    ({
      { -1.2247f, -1.2247f, -1.2247f },
      {  0.0f,     0.0f,     0.0f },
      {  1.2247f,  1.2247f,  1.2247f }
     });
  for (std::size_t m = 0; m < mat_f.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_f.get_col_count(); ++n)
    {
      auto cell1 = mat_f[{m, n}];
      auto cell2 = res_f[{m, n}];
      BOOST_CHECK_CLOSE(cell1, cell2, 0.01f);
    }
}


BOOST_AUTO_TEST_CASE(stddev_center_5x3)
{
  auto mat_d = mfmat::cl_mat_d
    ({
      { 90.0, 80.0, 40.0 },
      { 90.0, 60.0, 80.0 },
      { 60.0, 50.0, 70.0 },
      { 30.0, 40.0, 70.0 },
      { 30.0, 20.0, 90.0 }
     });
  mat_d.stddev_center();
  BOOST_CHECK_EQUAL(mat_d.get_row_count(), 5);
  BOOST_CHECK_EQUAL(mat_d.get_col_count(), 3);
  auto res_d = mfmat::cl_mat_d
    ({
      {  1.118,  1.5, -1.7928 },
      {  1.118,  0.5,  0.5976 },
      {  0.0  ,  0.0,  0.0    },
      { -1.118, -0.5,  0.0    },
      { -1.118, -1.5,  1.1952 }
     });
  for (std::size_t m = 0; m < mat_d.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_d.get_col_count(); ++n)
    {
      auto cell1 = mat_d[{m, n}];
      auto cell2 = res_d[{m, n}];
      BOOST_CHECK_CLOSE(cell1, cell2, 0.01);
    }
}


BOOST_AUTO_TEST_CASE(stddev_center_empty_matrix, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{0, 0};
  BOOST_CHECK_THROW(mat_f.stddev_center(), std::runtime_error);
  auto mat_d = mfmat::cl_mat_d{0, 0};
  BOOST_CHECK_THROW(mat_d.stddev_center(), std::runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
