#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/tools/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

namespace but = boost::unit_test;

BOOST_AUTO_TEST_SUITE(cl_mean_center_test_suite)

namespace
{

  void setup()
  {
    mfmat::cl_default_gpu_setter::instance();
    mfmat::cl_kernels_store::instance();
  }

} // !namespace anonymous

BOOST_AUTO_TEST_CASE(mean_center_identity, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{300, mfmat::identity_tag{}};
  mat_f.mean_center();
  BOOST_CHECK_EQUAL(mat_f.get_row_count(), 300);
  BOOST_CHECK_EQUAL(mat_f.get_col_count(), 300);
  for (std::size_t m = 0; m < mat_f.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_f.get_col_count(); ++n)
      if (m == n)
        BOOST_CHECK(mfmat::are_equal(mat_f.get(m, n), 299.0f / 300.0f));
      else
        BOOST_CHECK(mfmat::are_equal(mat_f.get(m, n), -1.0f / 300.0f));
  auto mat_d = mfmat::cl_mat_d{350, mfmat::identity_tag{}};
  mat_d.mean_center();
  BOOST_CHECK_EQUAL(mat_d.get_row_count(), 350);
  BOOST_CHECK_EQUAL(mat_d.get_col_count(), 350);
  for (std::size_t m = 0; m < mat_d.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_d.get_col_count(); ++n)
      if (m == n)
        BOOST_CHECK(mfmat::are_equal(mat_d.get(m, n), 349.0 / 350.0));
      else
        BOOST_CHECK(mfmat::are_equal(mat_d.get(m, n), -1.0 / 350.0));
}


BOOST_AUTO_TEST_CASE(mean_center, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{400, 500};
  for (std::size_t m = 0; m < mat_f.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_f.get_col_count(); ++n)
      mat_f[{ m, n }] = static_cast<float>(m + n);
  auto tmp_mean_f = mfmat::mean(mat_f);
  auto tmp_f = mat_f - tmp_mean_f;
  mat_f.mean_center();
  auto tmp_mean_zero_f = mfmat::mean(mat_f);
  BOOST_CHECK_EQUAL(mat_f.get_row_count(), 400);
  BOOST_CHECK_EQUAL(mat_f.get_col_count(), 500);
  for (std::size_t m = 0; m < mat_f.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_f.get_col_count(); ++n)
    {
      auto cell_res = mat_f.get(m, n);
      auto cell_tmp = tmp_f.get(m, n);
      BOOST_CHECK(mfmat::are_equal(cell_res, cell_tmp));
    }
  for (auto cell : tmp_mean_zero_f)
    BOOST_CHECK(mfmat::is_zero(cell));
  auto mat_d = mfmat::cl_mat_d{700, 600};
  for (std::size_t m = 0; m < mat_d.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_d.get_col_count(); ++n)
      mat_d[{ m, n }] = static_cast<double>(m + n);
  auto tmp_mean_d = mfmat::mean(mat_d);
  auto tmp_d = mat_d - tmp_mean_d;
  mat_d.mean_center();
  auto tmp_mean_zero_d = mfmat::mean(mat_d);
  BOOST_CHECK_EQUAL(mat_d.get_row_count(), 700);
  BOOST_CHECK_EQUAL(mat_d.get_col_count(), 600);
  for (std::size_t m = 0; m < mat_d.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_d.get_col_count(); ++n)
    {
      auto cell_res = mat_d.get(m, n);
      auto cell_tmp = tmp_d.get(m, n);
      BOOST_CHECK(mfmat::are_equal(cell_res, cell_tmp));
    }
  for (auto cell : tmp_mean_zero_d)
    BOOST_CHECK(mfmat::is_zero(cell));
}


BOOST_AUTO_TEST_CASE(mean_center_empty_matrix, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{0, 0};
  BOOST_CHECK_THROW(mat_f.mean_center(), std::runtime_error);
  auto mat_d = mfmat::cl_mat_d{0, 0};
  BOOST_CHECK_THROW(mat_d.mean_center(), std::runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
