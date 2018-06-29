#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

namespace but = boost::unit_test;

BOOST_AUTO_TEST_SUITE(cl_construction_test_suite)

namespace
{

  void setup()
  {
    mfmat::cl_default_gpu_setter::instance();
    mfmat::cl_kernels_store::instance();
  }

} // !namespace anonymous


BOOST_AUTO_TEST_CASE(rectangular_constructor, * but::fixture(&setup))
{
  auto mat = mfmat::cl_mat_f{10, 20};
  BOOST_CHECK_EQUAL(mat.get_row_count(), 10);
  BOOST_CHECK_EQUAL(mat.get_col_count(), 20);
  for (auto i : mat)
    BOOST_CHECK(mfmat::is_zero(i));
  for (std::size_t m = 0; m < mat.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat.get_col_count(); ++n)
    {
      auto cell1 = mat[{m, n}];
      auto cell2 = mat.get(m, n);
      BOOST_CHECK(mfmat::is_zero(cell1));
      BOOST_CHECK(mfmat::is_zero(cell2));
    }
}


BOOST_AUTO_TEST_CASE(square_constructor, * but::fixture(&setup))
{
  auto mat = mfmat::cl_mat_d{32};
  BOOST_CHECK_EQUAL(mat.get_row_count(), 32);
  BOOST_CHECK_EQUAL(mat.get_col_count(), 32);
  for (auto i : mat)
    BOOST_CHECK(mfmat::is_zero(i));
}


BOOST_AUTO_TEST_CASE(empty_matrix, * but::fixture(&setup))
{
  auto mat = mfmat::cl_mat_f{0};
  BOOST_CHECK_EQUAL(mat.get_row_count(), 0);
  BOOST_CHECK_EQUAL(mat.get_col_count(), 0);
  for (auto i : mat)
    throw std::range_error(std::to_string(i));
}


BOOST_AUTO_TEST_CASE(identity_constructor, * but::fixture(&setup))
{
  auto mat = mfmat::cl_mat_d{64, mfmat::identity_tag{}};
  BOOST_CHECK_EQUAL(mat.get_row_count(), 64);
  BOOST_CHECK_EQUAL(mat.get_col_count(), 64);
  for (std::size_t m = 0; m < mat.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat.get_col_count(); ++n)
    {
      auto cell1 = mat[{m, n}];
      auto cell2 = mat.get(m, n);
      if (m == n)
      {
        BOOST_CHECK_CLOSE(cell1, 1.0, 0.0000001);
        BOOST_CHECK_CLOSE(cell2, 1.0, 0.0000001);
      }
      else
      {
        BOOST_CHECK(mfmat::is_zero(cell1));
        BOOST_CHECK(mfmat::is_zero(cell2));
      }
    }
}


BOOST_AUTO_TEST_CASE(move_constructor, * but::fixture(&setup))
{
  auto mat_orig = mfmat::cl_mat_f{64};
  auto mat = std::move(mat_orig);
  BOOST_CHECK_EQUAL(mat_orig.get_row_count(), 0);
  BOOST_CHECK_EQUAL(mat_orig.get_col_count(), 0);
  BOOST_CHECK(mat_orig.begin() == mat_orig.end());
  BOOST_CHECK(mat_orig.cbegin() == mat_orig.cend());
  BOOST_CHECK_EQUAL(mat.get_row_count(), 64);
  BOOST_CHECK_EQUAL(mat.get_col_count(), 64);
  auto dist1 = std::distance(mat.begin(), mat.end());
  auto dist2 = std::distance(mat.cbegin(), mat.cend());
  BOOST_CHECK_EQUAL(dist1, 64 * 64);
  BOOST_CHECK_EQUAL(dist2, 64 * 64);
}


BOOST_AUTO_TEST_CASE(assignement_operator, * but::fixture(&setup))
{
  auto mat_orig = mfmat::cl_mat_d{64};
  auto mat = mfmat::cl_mat_d{32};
  mat = std::move(mat_orig);
  BOOST_CHECK_EQUAL(mat_orig.get_row_count(), 0);
  BOOST_CHECK_EQUAL(mat_orig.get_col_count(), 0);
  BOOST_CHECK(mat_orig.begin() == mat_orig.end());
  BOOST_CHECK(mat_orig.cbegin() == mat_orig.cend());
  BOOST_CHECK_EQUAL(mat.get_row_count(), 64);
  BOOST_CHECK_EQUAL(mat.get_col_count(), 64);
  auto dist1 = std::distance(mat.begin(), mat.end());
  auto dist2 = std::distance(mat.cbegin(), mat.cend());
  BOOST_CHECK_EQUAL(dist1, 64 * 64);
  BOOST_CHECK_EQUAL(dist2, 64 * 64);
}

BOOST_AUTO_TEST_SUITE_END()
