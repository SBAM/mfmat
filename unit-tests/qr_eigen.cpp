#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(QR_eigen_test_suite)

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


BOOST_AUTO_TEST_CASE(qr_eigen_2x2_1)
{
  auto mat = mfmat::ct_mat<double, 2, 2>
    ({
      {  2.0, 3.0 },
      {  3.0, 4.0 }
     });
  auto values = mfmat::ct_mat<double, 2, 2>
    ({
      { 6.1623,  0.0000 },
      { 0.0000, -0.1623  }
     });
  auto vectors = mfmat::ct_mat<double, 2, 2>
    ({
      { 0.5847, -0.8112 },
      { 0.8112,  0.5847 }
     });
  auto qre = mfmat::qr_eigen(mat);
  check_tolerance(qre.get_values(), values, 1.0e-4);
  check_tolerance(qre.get_vectors(), vectors, 1.0e-4);
  qre(mat);
  check_tolerance(qre.get_values(), values, 1.0e-4);
  check_tolerance(qre.get_vectors(), vectors, 1.0e-4);
}


BOOST_AUTO_TEST_CASE(qr_eigen_4x4_1)
{
  auto mat = mfmat::ct_mat<double, 4, 4>
    ({
      { 5.0, 0.0, 0.0, 0.0 },
      { 0.0, 4.0, 0.0, 0.0 },
      { 0.0, 0.0, 3.0, 0.0 },
      { 0.0, 0.0, 0.0, 2.0 }
     });
  auto values = mfmat::ct_mat<double, 4, 4>
    ({
      { 5.0, 0.0, 0.0, 0.0 },
      { 0.0, 4.0, 0.0, 0.0 },
      { 0.0, 0.0, 3.0, 0.0 },
      { 0.0, 0.0, 0.0, 2.0 }
     });
  auto vectors = mfmat::ct_mat<double, 4, 4>(mfmat::identity_tag{});
  auto qre = mfmat::qr_eigen(mat);
  BOOST_CHECK(qre.get_values() == values);
  BOOST_CHECK(qre.get_vectors() == vectors);
  qre(mat);
  BOOST_CHECK(qre.get_values() == values);
  BOOST_CHECK(qre.get_vectors() == vectors);
}


BOOST_AUTO_TEST_CASE(qr_eigen_4x4_2)
{
  auto mat = mfmat::ct_mat<double, 4, 4>
    ({
      { 52.0, 30.0, 49.0, 28.0 },
      { 30.0, 50.0,  8.0, 44.0 },
      { 49.0,  8.0, 46.0, 16.0 },
      { 28.0, 44.0, 16.0, 22.0 }
     });
  auto values = mfmat::ct_mat<double, 4, 4>
    ({
      { 132.6279,  0.0000,   0.00000,  0.00000 },
      {   0.0000, 52.4423,   0.00000,  0.00000 },
      {   0.0000,  0.0000, -11.54113,  0.00000 },
      {   0.0000,  0.0000,   0.00000, -3.52904 }
     });
  auto vectors = mfmat::ct_mat<double, 4, 4>
    ({
      { 0.60946, -0.29992,  0.09988,  0.72707 },
      { 0.48785,  0.65200, -0.57725, -0.06069 },
      { 0.46658, -0.60196, -0.22156, -0.60898 },
      { 0.41577,  0.35013,  0.77956, -0.31117 }
     });
  auto qre = mfmat::qr_eigen(mat);
  check_tolerance(qre.get_values(), values, 1.0e-4);
  check_tolerance(qre.get_vectors(), vectors, 1.0e-4);
  qre(mat, 512);
  check_tolerance(qre.get_values(), values, 1.0e-4);
  check_tolerance(qre.get_vectors(), vectors, 1.0e-4);
}


BOOST_AUTO_TEST_CASE(qr_eigen_5x5_1)
{
  auto mat = mfmat::ct_mat<double, 5, 5>
    ({
      { 1.0 / 1.0, 1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0, 1.0 / 5.0 },
      { 1.0 / 2.0, 2.0 / 2.0, 2.0 / 3.0, 2.0 / 4.0, 2.0 / 5.0 },
      { 1.0 / 3.0, 2.0 / 3.0, 3.0 / 3.0, 3.0 / 4.0, 3.0 / 5.0 },
      { 1.0 / 4.0, 2.0 / 4.0, 3.0 / 4.0, 4.0 / 4.0, 4.0 / 5.0 },
      { 1.0 / 5.0, 2.0 / 5.0, 3.0 / 5.0, 4.0 / 5.0, 5.0 / 5.0 }
     });
  auto values = mfmat::ct_mat<double, 5, 5>
    ({
      { 3.0666, 0.0000, 0.0000, 0.0000, 0.0000 },
      { 0.0000, 1.0035, 0.0000, 0.0000, 0.0000 },
      { 0.0000, 0.0000, 0.5010, 0.0000, 0.0000 },
      { 0.0000, 0.0000, 0.0000, 0.2728, 0.0000 },
      { 0.0000, 0.0000, 0.0000, 0.0000, 0.1560 }
     });
  auto vectors = mfmat::ct_mat<double, 5, 5>
    ({
      { 0.2939, -0.7510,  0.5681, -0.1628,  0.0164 },
      { 0.4441, -0.3795, -0.5646,  0.5695, -0.1252 },
      { 0.5056,  0.0578, -0.3803, -0.6368,  0.4370 },
      { 0.5029,  0.3360,  0.1535, -0.1832, -0.7596 },
      { 0.4559,  0.4192,  0.4362,  0.4584,  0.4648 }
     });
  auto qre = mfmat::qr_eigen(mat);
  check_tolerance(qre.get_values(), values, 1.0e-4);
  check_tolerance(qre.get_vectors(), vectors, 1.0e-4);
  qre(mat, 64);
  check_tolerance(qre.get_values(), values, 1.0e-4);
  check_tolerance(qre.get_vectors(), vectors, 1.0e-4);
}
BOOST_AUTO_TEST_SUITE_END()
