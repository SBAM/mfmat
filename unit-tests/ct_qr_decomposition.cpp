#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/tools/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(ct_QR_decomposition_test_suite)

template <typename T>
void check_tolerance(const T& lhs, const T& rhs, double tolerance)
{
  for (std::size_t r = 0; r < lhs.row_count; ++r)
    for (std::size_t c = 0; c < lhs.col_count; ++c)
    {
      auto cell_1 = lhs[{r, c}];
      auto cell_2 = rhs[{r, c}];
      BOOST_CHECK_SMALL(cell_1 - cell_2, tolerance);
    }
}


BOOST_AUTO_TEST_CASE(qr_decomposition_3x3_1)
{
  auto mat = mfmat::ct_mat<double, 3, 3>
    ({
      {  12.0, -51.0,   4.0 },
      {   6.0, 167.0, -68.0 },
      {  -4.0,  24.0, -41.0 }
     });
  auto q_res = mfmat::ct_mat<double, 3, 3>
    ({
      {  6.0 / 7.0, -69.0 / 175.0, -58.0 / 175.0 },
      {  3.0 / 7.0, 158.0 / 175.0,   6.0 / 175.0 },
      { -2.0 / 7.0,   6.0 /  35.0, -33.0 /  35.0 }
     });
  auto r_res = mfmat::ct_mat<double, 3, 3>
    ({
      { 14.0,  21.0, -14.0 },
      {  0.0, 175.0, -70.0 },
      {  0.0,   0.0,  35.0 }
     });
  auto qrd = mfmat::qr_decomposition(mat);
  check_tolerance(qrd.get_q(), q_res, 1.0e-14);
  check_tolerance(qrd.get_r(), r_res, 3.0e-14);
  qrd(mat);
  check_tolerance(qrd.get_q(), q_res, 1.0e-14);
  check_tolerance(qrd.get_r(), r_res, 3.0e-14);
  auto qr_res = qrd.get_q() * qrd.get_r();
  check_tolerance(qr_res, mat, 3.0e-14);
}


BOOST_AUTO_TEST_CASE(qr_decomposition_3x3_2)
{
  auto mat = mfmat::ct_mat<double, 3, 3>
    ({
      { 0.0, 1.0, 1.0 },
      { 1.0, 1.0, 2.0 },
      { 0.0, 0.0, 3.0 }
     });
  auto q_res = mfmat::ct_mat<double, 3, 3>
    ({
      { 0.0, 1.0, 0.0 },
      { 1.0, 0.0, 0.0 },
      { 0.0, 0.0, 1.0 }
     });
  auto r_res = mfmat::ct_mat<double, 3, 3>
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


BOOST_AUTO_TEST_CASE(qr_decomposition_4x3_1)
{
  auto mat = mfmat::ct_mat<double, 4, 3>
    ({
      { 1.0,  0.0, -1.0 },
      { 1.0,  0.0, -3.0 },
      { 0.0,  1.0,  1.0 },
      { 0.0, -1.0,  1.0 },
     });
  auto q_res = mfmat::ct_mat<double, 4, 3>
    ({
      { 1.0 / std::sqrt(2.0), 0.0, 0.5 },
      { 1.0 / std::sqrt(2.0), 0.0, -0.5 },
      { 0.0, 1.0 / std::sqrt(2.0), 0.5 },
      { 0.0, -1.0 / std::sqrt(2.0), 0.5 }
     });
  auto r_res = mfmat::ct_mat<double, 3, 3>
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


BOOST_AUTO_TEST_CASE(qr_decomposition_3x5_1)
{
  auto mat = mfmat::ct_mat<double, 3, 5>
    ({
      { -1.0, 9.0,  2.0, 8.0,  7.0 },
      {  5.0, 6.0, -5.0, 7.0,  2.0 },
      { -9.0, 0.0,  1.0, 2.0, -3.0 }
     });
  auto qrd = mfmat::qr_decomposition(mat);
  auto qr_res = qrd.get_q() * qrd.get_r();
  check_tolerance(qr_res, mat, 1.0e-14);
}


BOOST_AUTO_TEST_CASE(qr_decomposition_5x3_1)
{
  auto mat = mfmat::ct_mat<double, 5, 3>
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
