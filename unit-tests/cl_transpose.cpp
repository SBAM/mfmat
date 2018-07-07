#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

namespace but = boost::unit_test;

BOOST_AUTO_TEST_SUITE(cl_transpose_test_suite)

namespace
{

  void setup()
  {
    mfmat::cl_default_gpu_setter::instance();
    mfmat::cl_kernels_store::instance();
  }

} // !namespace anonymous

BOOST_AUTO_TEST_CASE(transpose_in_place_square, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{100, 100};
  std::size_t idx = 0;
  for (auto& cell : mat_f)
    cell = static_cast<float>(idx++);
  mat_f.transpose();
  idx = 0;
  for (std::size_t n = 0; n < mat_f.get_col_count(); ++n)
    for (std::size_t m = 0; m < mat_f.get_row_count(); ++m)
    {
      auto cell = mat_f.get(m, n);
      BOOST_CHECK(mfmat::are_equal(cell, static_cast<float>(idx++)));
    }
  auto mat_d = mfmat::cl_mat_d{150, 150};
  idx = 0;
  for (auto& cell : mat_d)
    cell = static_cast<double>(idx++);
  mat_d.transpose();
  idx = 0;
  for (std::size_t n = 0; n < mat_d.get_col_count(); ++n)
    for (std::size_t m = 0; m < mat_d.get_row_count(); ++m)
    {
      auto cell = mat_d.get(m, n);
      BOOST_CHECK(mfmat::are_equal(cell, static_cast<double>(idx++)));
    }
}


BOOST_AUTO_TEST_CASE(transpose_in_place_rectangular, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{100, 200};
  std::size_t idx = 0;
  for (auto& cell : mat_f)
    cell = static_cast<float>(idx++);
  mat_f.transpose();
  BOOST_CHECK_EQUAL(mat_f.get_row_count(), 200);
  BOOST_CHECK_EQUAL(mat_f.get_col_count(), 100);
  idx = 0;
  for (std::size_t n = 0; n < mat_f.get_col_count(); ++n)
    for (std::size_t m = 0; m < mat_f.get_row_count(); ++m)
    {
      auto cell = mat_f.get(m, n);
      BOOST_CHECK(mfmat::are_equal(cell, static_cast<float>(idx++)));
    }
  auto mat_d = mfmat::cl_mat_d{200, 150};
  idx = 0;
  for (auto& cell : mat_d)
    cell = static_cast<double>(idx++);
  mat_d.transpose();
  BOOST_CHECK_EQUAL(mat_d.get_row_count(), 150);
  BOOST_CHECK_EQUAL(mat_d.get_col_count(), 200);
  idx = 0;
  for (std::size_t n = 0; n < mat_d.get_col_count(); ++n)
    for (std::size_t m = 0; m < mat_d.get_row_count(); ++m)
    {
      auto cell = mat_d.get(m, n);
      BOOST_CHECK(mfmat::are_equal(cell, static_cast<double>(idx++)));
    }
}


BOOST_AUTO_TEST_CASE(transpose_square, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{150, 150};
  std::size_t idx = 0;
  for (auto& cell : mat_f)
    cell = static_cast<float>(idx++);
  auto res_f = mfmat::transpose(mat_f);
  idx = 0;
  for (std::size_t n = 0; n < res_f.get_col_count(); ++n)
    for (std::size_t m = 0; m < res_f.get_row_count(); ++m)
    {
      auto cell = res_f.get(m, n);
      BOOST_CHECK(mfmat::are_equal(cell, static_cast<float>(idx++)));
    }
  auto mat_d = mfmat::cl_mat_d{175, 175};
  idx = 0;
  for (auto& cell : mat_d)
    cell = static_cast<double>(idx++);
  auto res_d = mfmat::transpose(mat_d);
  idx = 0;
  for (std::size_t n = 0; n < res_d.get_col_count(); ++n)
    for (std::size_t m = 0; m < res_d.get_row_count(); ++m)
    {
      auto cell = res_d.get(m, n);
      BOOST_CHECK(mfmat::are_equal(cell, static_cast<double>(idx++)));
    }
}


BOOST_AUTO_TEST_CASE(transpose_rectangular, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{250, 200};
  std::size_t idx = 0;
  for (auto& cell : mat_f)
    cell = static_cast<float>(idx++);
  auto res_f = mfmat::transpose(mat_f);
  BOOST_CHECK_EQUAL(res_f.get_row_count(), 200);
  BOOST_CHECK_EQUAL(res_f.get_col_count(), 250);
  idx = 0;
  for (std::size_t n = 0; n < res_f.get_col_count(); ++n)
    for (std::size_t m = 0; m < res_f.get_row_count(); ++m)
    {
      auto cell = res_f.get(m, n);
      BOOST_CHECK(mfmat::are_equal(cell, static_cast<float>(idx++)));
    }
  auto mat_d = mfmat::cl_mat_d{250, 300};
  idx = 0;
  for (auto& cell : mat_d)
    cell = static_cast<double>(idx++);
  auto res_d = mfmat::transpose(mat_d);
  BOOST_CHECK_EQUAL(res_d.get_row_count(), 300);
  BOOST_CHECK_EQUAL(res_d.get_col_count(), 250);
  idx = 0;
  for (std::size_t n = 0; n < res_d.get_col_count(); ++n)
    for (std::size_t m = 0; m < res_d.get_row_count(); ++m)
    {
      auto cell = res_d.get(m, n);
      BOOST_CHECK(mfmat::are_equal(cell, static_cast<double>(idx++)));
    }
}

BOOST_AUTO_TEST_SUITE_END()
