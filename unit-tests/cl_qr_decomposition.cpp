#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/tools/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

namespace but = boost::unit_test;

BOOST_AUTO_TEST_SUITE(cl_QR_decomposition_test_suite)

namespace
{

  void setup()
  {
    mfmat::cl_default_gpu_setter::instance();
    mfmat::cl_kernels_store::instance();
  }

} // !namespace anonymous

template <typename T>
void check_tolerance(const mfmat::cl_mat<T>& lhs,
                     const mfmat::cl_mat<T>& rhs,
                     double tolerance)
{
  BOOST_CHECK_EQUAL(lhs.get_row_count(), rhs.get_row_count());
  BOOST_CHECK_EQUAL(lhs.get_col_count(), rhs.get_col_count());
  for (std::size_t r = 0; r < lhs.get_row_count(); ++r)
    for (std::size_t c = 0; c < lhs.get_col_count(); ++c)
    {
      auto cell_1 = lhs.get(r, c);
      auto cell_2 = rhs.get(r, c);
      BOOST_CHECK_SMALL(cell_1 - cell_2, tolerance);
    }
}


BOOST_AUTO_TEST_CASE(qr_decomposition_3x3_1, * but::fixture(&setup))
{
  auto mat = mfmat::cl_mat_d
    ({
      {  12.0, -51.0,   4.0 },
      {   6.0, 167.0, -68.0 },
      {  -4.0,  24.0, -41.0 }
     });
  auto q_res = mfmat::cl_mat_d
    ({
      {  6.0 / 7.0, -69.0 / 175.0, -58.0 / 175.0 },
      {  3.0 / 7.0, 158.0 / 175.0,   6.0 / 175.0 },
      { -2.0 / 7.0,   6.0 /  35.0, -33.0 /  35.0 }
     });
  auto r_res = mfmat::cl_mat_d
    ({
      { 14.0,  21.0, -14.0 },
      {  0.0, 175.0, -70.0 },
      {  0.0,   0.0,  35.0 }
     });
  auto qrd = mfmat::qr_decomposition(mat);
  check_tolerance(qrd.get_q(), q_res, 1.0e-13);
  check_tolerance(qrd.get_r(), r_res, 1.0e-13);
  qrd(mat);
  check_tolerance(qrd.get_q(), q_res, 1.0e-13);
  check_tolerance(qrd.get_r(), r_res, 1.0e-13);
  auto qr_res = qrd.get_q() * qrd.get_r();
  check_tolerance(qr_res, mat, 1.0e-13);
}


BOOST_AUTO_TEST_CASE(qr_decomposition_3x3_2, * but::fixture(&setup))
{
  auto mat = mfmat::cl_mat_d
    ({
      { 0.0, 1.0, 1.0 },
      { 1.0, 1.0, 2.0 },
      { 0.0, 0.0, 3.0 }
     });
  auto q_res = mfmat::cl_mat_d
    ({
      { 0.0, 1.0, 0.0 },
      { 1.0, 0.0, 0.0 },
      { 0.0, 0.0, 1.0 }
     });
  auto r_res = mfmat::cl_mat_d
    ({
      { 1.0, 1.0, 2.0 },
      { 0.0, 1.0, 1.0 },
      { 0.0, 0.0, 3.0 }
     });
  auto qrd = mfmat::qr_decomposition(mat);
  BOOST_CHECK(qrd.get_q() == q_res);
  BOOST_CHECK(qrd.get_r() == r_res);
  qrd(mat);
  BOOST_CHECK(qrd.get_q() == q_res);
  BOOST_CHECK(qrd.get_r() == r_res);
  auto qr_res = qrd.get_q() * qrd.get_r();
  BOOST_CHECK(qr_res == mat);
}


BOOST_AUTO_TEST_CASE(qr_decomposition_4x3_1, * but::fixture(&setup))
{
  auto mat = mfmat::cl_mat_d
    ({
      { 1.0,  0.0, -1.0 },
      { 1.0,  0.0, -3.0 },
      { 0.0,  1.0,  1.0 },
      { 0.0, -1.0,  1.0 },
     });
  auto q_res = mfmat::cl_mat_d
    ({
      { 1.0 / std::sqrt(2.0), 0.0, 0.5 },
      { 1.0 / std::sqrt(2.0), 0.0, -0.5 },
      { 0.0, 1.0 / std::sqrt(2.0), 0.5 },
      { 0.0, -1.0 / std::sqrt(2.0), 0.5 }
     });
  auto r_res = mfmat::cl_mat_d
    ({
      { std::sqrt(2.0), 0.0, -2.0 * std::sqrt(2.0) },
      { 0.0, std::sqrt(2.0), 0.0 },
      { 0.0, 0.0, 2.0 }
     });
  auto qrd = mfmat::qr_decomposition(mat);
  check_tolerance(qrd.get_q(), q_res, 1.0e-15);
  check_tolerance(qrd.get_r(), r_res, 1.0e-15);
  qrd(mat);
  check_tolerance(qrd.get_q(), q_res, 1.0e-15);
  check_tolerance(qrd.get_r(), r_res, 1.0e-15);
  auto qr_res = qrd.get_q() * qrd.get_r();
  check_tolerance(qr_res, mat, 1.0e-15);
}


BOOST_AUTO_TEST_CASE(qr_decomposition_3x5_1, * but::fixture(&setup))
{
  auto mat = mfmat::cl_mat_d
    ({
      { -1.0, 9.0,  2.0, 8.0,  7.0 },
      {  5.0, 6.0, -5.0, 7.0,  2.0 },
      { -9.0, 0.0,  1.0, 2.0, -3.0 }
     });
  auto qrd = mfmat::qr_decomposition(mat);
  auto qr_res = qrd.get_q() * qrd.get_r();
  check_tolerance(qr_res, mat, 1.0e-14);
}


BOOST_AUTO_TEST_CASE(qr_decomposition_5x3_1, * but::fixture(&setup))
{
  auto mat = mfmat::cl_mat_d
    ({
      { -1.0,  9.0,  2.0 },
      {  8.0,  7.0,  5.0 },
      {  6.0, -5.0,  7.0 },
      {  2.0, -9.0,  0.0 },
      {  1.0,  2.0, -3.0 }
     });
  auto qrd = mfmat::qr_decomposition(mat);
  auto qr_res = qrd.get_q() * qrd.get_r();
  check_tolerance(qr_res, mat, 2.0e-15);
}

BOOST_AUTO_TEST_SUITE_END()
