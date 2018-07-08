#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

namespace but = boost::unit_test;

BOOST_AUTO_TEST_SUITE(cl_matrices_multiply_suite)

namespace
{

  void setup()
  {
    mfmat::cl_default_gpu_setter::instance();
    mfmat::cl_kernels_store::instance();
  }

} // !namespace anonymous

BOOST_AUTO_TEST_CASE(multiply_identity, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{300, mfmat::identity_tag{}};
  auto res_f = mat_f * mat_f;
  for (std::size_t m = 0; m < res_f.get_row_count(); ++m)
    for (std::size_t n = 0; n < res_f.get_col_count(); ++n)
    {
      auto cell = res_f.get(m, n);
      if (m == n)
        BOOST_CHECK(mfmat::are_equal(cell, 1.0f));
      else
        BOOST_CHECK(mfmat::is_zero(cell));
    }
  auto mat_d = mfmat::cl_mat_d{350, mfmat::identity_tag{}};
  auto res_d = mat_d * mat_d;
  for (std::size_t m = 0; m < res_d.get_row_count(); ++m)
    for (std::size_t n = 0; n < res_d.get_col_count(); ++n)
    {
      auto cell = res_d.get(m, n);
      if (m == n)
        BOOST_CHECK(mfmat::are_equal(cell, 1.0));
      else
        BOOST_CHECK(mfmat::is_zero(cell));
    }
}


BOOST_AUTO_TEST_CASE(multiply_ones, * but::fixture(&setup))
{
  auto mat_f_lhs = mfmat::cl_mat_f{400, 500};
  for (auto& cell : mat_f_lhs)
    cell = 1.0f;
  auto mat_f_rhs = mfmat::cl_mat_f{500, 600};
  for (auto& cell : mat_f_rhs)
    cell = 1.0f;
  auto res_f = mat_f_lhs * mat_f_rhs;
  BOOST_CHECK_EQUAL(res_f.get_row_count(), 400);
  BOOST_CHECK_EQUAL(res_f.get_col_count(), 600);
  for (auto& cell : res_f)
    BOOST_CHECK(mfmat::are_equal(cell, 500.0f));
  auto mat_d_lhs = mfmat::cl_mat_d{450, 550};
  for (auto& cell : mat_d_lhs)
    cell = 1.0;
  auto mat_d_rhs = mfmat::cl_mat_d{550, 650};
  for (auto& cell : mat_d_rhs)
    cell = 1.0;
  auto res_d = mat_d_lhs * mat_d_rhs;
  BOOST_CHECK_EQUAL(res_d.get_row_count(), 450);
  BOOST_CHECK_EQUAL(res_d.get_col_count(), 650);
  for (auto& cell : res_d)
    BOOST_CHECK(mfmat::are_equal(cell, 550.0));
}


BOOST_AUTO_TEST_CASE(multiply_outer, * but::fixture(&setup))
{
  auto mat_f_lhs = mfmat::cl_mat_f{400, 10};
  for (auto& cell : mat_f_lhs)
    cell = 2.0f;
  auto mat_f_rhs = mfmat::cl_mat_f{10, 300};
  for (auto& cell : mat_f_rhs)
    cell = 2.0f;
  auto res_f = mat_f_lhs * mat_f_rhs;
  BOOST_CHECK_EQUAL(res_f.get_row_count(), 400);
  BOOST_CHECK_EQUAL(res_f.get_col_count(), 300);
  for (auto& cell : res_f)
    BOOST_CHECK(mfmat::are_equal(cell, 40.0f));
  auto mat_d_lhs = mfmat::cl_mat_d{450, 15};
  for (auto& cell : mat_d_lhs)
    cell = 2.0;
  auto mat_d_rhs = mfmat::cl_mat_d{15, 350};
  for (auto& cell : mat_d_rhs)
    cell = 2.0;
  auto res_d = mat_d_lhs * mat_d_rhs;
  BOOST_CHECK_EQUAL(res_d.get_row_count(), 450);
  BOOST_CHECK_EQUAL(res_d.get_col_count(), 350);
  for (auto& cell : res_d)
    BOOST_CHECK(mfmat::are_equal(cell, 60.0));
}


BOOST_AUTO_TEST_CASE(multiply_alternate, * but::fixture(&setup))
{
  auto mat_f_lhs = mfmat::cl_mat_f{300, 350};
  for (auto& cell : mat_f_lhs)
    cell = 1.0f;
  auto mat_f_rhs = mfmat::cl_mat_f{350, 400};
  for (std::size_t m = 0; m < mat_f_rhs.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_f_rhs.get_col_count(); ++n)
      if (m % 2 == 0)
        mat_f_rhs.get(m, n) = 1.0f;
      else
        mat_f_rhs.get(m, n) = -1.0f;
  auto res_f = mat_f_lhs * mat_f_rhs;
  BOOST_CHECK_EQUAL(res_f.get_row_count(), 300);
  BOOST_CHECK_EQUAL(res_f.get_col_count(), 400);
  for (auto& cell : res_f)
    BOOST_CHECK(mfmat::is_zero(cell));
  auto mat_d_lhs = mfmat::cl_mat_d{450, 400};
  for (auto& cell : mat_d_lhs)
    cell = 1.0;
  auto mat_d_rhs = mfmat::cl_mat_d{400, 350};
  for (std::size_t m = 0; m < mat_d_rhs.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_d_rhs.get_col_count(); ++n)
      if (m % 2 == 0)
        mat_d_rhs.get(m, n) = 1.0f;
      else
        mat_d_rhs.get(m, n) = -1.0f;
  auto res_d = mat_d_lhs * mat_d_rhs;
  BOOST_CHECK_EQUAL(res_d.get_row_count(), 450);
  BOOST_CHECK_EQUAL(res_d.get_col_count(), 350);
  for (auto& cell : res_d)
    BOOST_CHECK(mfmat::is_zero(cell));
}


BOOST_AUTO_TEST_CASE(incompatible_dimensions, * but::fixture(&setup))
{
  auto mat_f1 = mfmat::cl_mat_f{32, 32};
  auto mat_f2 = mfmat::cl_mat_f{33, 32};
  BOOST_CHECK_THROW(mat_f1 * mat_f2, std::runtime_error);
  auto mat_d1 = mfmat::cl_mat_f{33, 33};
  auto mat_d2 = mfmat::cl_mat_f{32, 33};
  BOOST_CHECK_THROW(mat_d1 * mat_d2, std::runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
