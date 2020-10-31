#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/tools/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

namespace but = boost::unit_test;

BOOST_AUTO_TEST_SUITE(cl_mean_test_suite)

namespace
{

  void setup()
  {
    mfmat::cl_default_gpu_setter::instance();
    mfmat::cl_kernels_store::instance();
  }

} // !namespace anonymous

BOOST_AUTO_TEST_CASE(mean_identity, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{300, mfmat::identity_tag{}};
  auto res_f = mfmat::mean(mat_f);
  BOOST_CHECK_EQUAL(res_f.get_row_count(), 1);
  BOOST_CHECK_EQUAL(res_f.get_col_count(), 300);
  for (auto cell : res_f)
    BOOST_CHECK(mfmat::are_equal(cell, 1.0f / 300.0f));
  auto mat_d = mfmat::cl_mat_d{350, mfmat::identity_tag{}};
  auto res_d = mfmat::mean(mat_d);
  BOOST_CHECK_EQUAL(res_d.get_row_count(), 1);
  BOOST_CHECK_EQUAL(res_d.get_col_count(), 350);
  for (auto cell : res_d)
    BOOST_CHECK(mfmat::are_equal(cell, 1.0 / 350.0));
}


BOOST_AUTO_TEST_CASE(mean, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{400, 500};
  for (std::size_t m = 0; m < mat_f.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_f.get_col_count(); ++n)
      mat_f[{ m, n }] = static_cast<float>(m + n);
  auto res_f = mfmat::mean(mat_f);
  BOOST_CHECK_EQUAL(res_f.get_row_count(), 1);
  BOOST_CHECK_EQUAL(res_f.get_col_count(), 500);
  for (std::size_t i = 0; i < res_f.get_col_count(); ++i)
  {
    auto cell = res_f.get(0, i);
    auto res = (mat_f.get(0, i) +
                mat_f.get(mat_f.get_row_count() - 1, i)) / 2.0f;
    BOOST_CHECK(mfmat::are_equal(cell, res));
  }
  auto mat_d = mfmat::cl_mat_d{700, 600};
  for (std::size_t m = 0; m < mat_d.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_d.get_col_count(); ++n)
      mat_d[{ m, n }] = static_cast<double>(m + n);
  auto res_d = mfmat::mean(mat_d);
  BOOST_CHECK_EQUAL(res_d.get_row_count(), 1);
  BOOST_CHECK_EQUAL(res_d.get_col_count(), 600);
  for (std::size_t i = 0; i < res_d.get_col_count(); ++i)
  {
    auto cell = res_d.get(0, i);
    auto res = (mat_d.get(0, i) +
                mat_d.get(mat_d.get_row_count() - 1, i)) / 2.0;
    BOOST_CHECK(mfmat::are_equal(cell, res));
  }
}


BOOST_AUTO_TEST_CASE(mean_empty_matrix, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{0, 0};
  BOOST_CHECK_THROW(mfmat::mean(mat_f), std::runtime_error);
  auto mat_d = mfmat::cl_mat_d{0, 0};
  BOOST_CHECK_THROW(mfmat::mean(mat_d), std::runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
