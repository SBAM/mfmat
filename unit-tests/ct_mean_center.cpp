#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/tools/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(ct_mean_center_test_suite)

BOOST_AUTO_TEST_CASE(identity_1x1_mean_center)
{
  auto mat1 = mfmat::ct_mat<std::int32_t, 1, 1>(mfmat::identity_tag());
  auto res1 = mfmat::ct_mat<std::int32_t, 1, 1>();
  auto mc11 = mat1;
  mc11.mean_center();
  auto mean1 = std::make_optional(mfmat::mean(mat1));
  auto mc12 = mat1;
  mc12.mean_center(mean1);
  BOOST_CHECK(res1 == mc11);
  BOOST_CHECK(res1 == mc12);
  auto mat2 = mfmat::ct_mat<double, 1, 1>({{2.0}});
  auto res2 = mfmat::ct_mat<double, 1, 1>();
  auto mc21 = mat2;
  mc21.mean_center();
  auto mean2 = std::make_optional(mfmat::mean(mat2));
  auto mc22 = mat2;
  mc22.mean_center(mean2);
  BOOST_CHECK(res2 == mc21);
  BOOST_CHECK(res2 == mc22);
}


BOOST_AUTO_TEST_CASE(mean_center_3x3)
{
  auto mat = mfmat::ct_mat<double, 3, 3>
    ({
      { 1.0, 2.0, 3.0 },
      { 4.0, 5.0, 6.0 },
      { 7.0, 8.0, 9.0 }
     });
  auto mc1 = mat;
  mc1.mean_center();
  auto mean = std::make_optional(mfmat::mean(mat));
  auto mc2 = mat;
  mc2.mean_center(mean);
  auto res = mfmat::ct_mat<double, 3, 3>
    ({
      { -3.0, -3.0, -3.0 },
      {  0.0,  0.0,  0.0 },
      {  3.0,  3.0,  3.0 }
     });
  BOOST_CHECK(res == mc1);
  BOOST_CHECK(res == mc2);
}


BOOST_AUTO_TEST_CASE(mean_center_5x3)
{
  auto mat = mfmat::ct_mat<double, 5, 3>
    ({
      { 90.0, 80.0, 40.0 },
      { 90.0, 60.0, 80.0 },
      { 60.0, 50.0, 70.0 },
      { 30.0, 40.0, 70.0 },
      { 30.0, 20.0, 90.0 }
     });
  auto mc1 = mat;
  mc1.mean_center();
  auto mean = std::make_optional(mfmat::mean(mat));
  auto mc2 = mat;
  mc2.mean_center(mean);
  auto res = mfmat::ct_mat<double, 5, 3>
    ({
      {  30.0,  30.0, -30.0 },
      {  30.0,  10.0,  10.0 },
      {   0.0,   0.0,   0.0 },
      { -30.0, -10.0,   0.0 },
      { -30.0, -30.0,  20.0 }
     });
  BOOST_CHECK(res == mc1);
  BOOST_CHECK(res == mc2);
}

BOOST_AUTO_TEST_SUITE_END()
