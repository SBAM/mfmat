#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(determinant_test_suite)

BOOST_AUTO_TEST_CASE(rec_det_2x2)
{
  auto mat1 = mfmat::dense_matrix<std::int32_t, 2, 2>(mfmat::identity_tag{});
  auto res1 = mat1.rec_det();
  BOOST_CHECK(res1 == 1);
  auto mat2 = mfmat::dense_matrix<std::int32_t, 2, 2>
    ({
      { 1, 2 },
      { 3, 4 }
     });
  auto res2 = mat2.rec_det();
  BOOST_CHECK(res2 == -2);
  auto mat3 = mfmat::dense_matrix<std::int32_t, 2, 2>();
  auto res3 = mat3.rec_det();
  BOOST_CHECK(res3 == 0);
}


BOOST_AUTO_TEST_CASE(rec_det_3x3)
{
  auto mat = mfmat::dense_matrix<std::int32_t, 3, 3>
    ({
      { -2,  2, -3 },
      { -1,  1,  3 },
      {  2,  0, -1 }
     });
  auto res = mat.rec_det();
  BOOST_CHECK(res == 18);
}


BOOST_AUTO_TEST_CASE(rec_det_5x5)
{
  auto mat = mfmat::dense_matrix<double, 5, 5>
    ({
      {  0.5,  0.6,  0.7,  0.8,  0.9 },
      { -0.1, -0.3, -0.2, -0.4, -0.5 },
      {  2.2, -3.3,  4.4, -5.5,  6.6 },
      {  0.0,  2.5,  0.0, -2.5,  0.0 },
      {  1.1,  1.2, -1.1, -1.2,  0.1 }
     });
  auto res = mat.rec_det();
  BOOST_CHECK_CLOSE(res, 17.061, 0.001);
}


BOOST_AUTO_TEST_CASE(rec_det_5x5_2)
{
  auto mat = mfmat::dense_matrix<std::int32_t, 5, 5>
    ({
      {   2,   3,   5,   7,  11 },
      {  13,  17,  19,  23,  31 },
      {  37,  39,  43, - 2, - 3 },
      { - 5, - 7, -11, -13, -17 },
      { -19, -23, -31, -37, -39 }
     });
  auto res = mat.rec_det();
  BOOST_CHECK(res == -11700);
}

BOOST_AUTO_TEST_SUITE_END()
