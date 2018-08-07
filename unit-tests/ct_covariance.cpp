#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(ct_covariance_test_suite)

BOOST_AUTO_TEST_CASE(identity_1x1_covariance)
{
  auto mat1 = mfmat::ct_mat<std::int32_t, 1, 1>(mfmat::identity_tag());
  auto res1 = mfmat::ct_mat<std::int32_t, 1, 1>();
  auto cov11 = mfmat::covariance(mat1);
  auto mean1 = std::make_optional(mfmat::mean(mat1));
  auto cov12 = mfmat::covariance(mat1, mean1);
  BOOST_CHECK(cov11 == res1);
  BOOST_CHECK(cov12 == res1);
  auto mat2 = mfmat::ct_mat<double, 1, 1>({{2.0}});
  auto res2 = mfmat::ct_mat<double, 1, 1>();
  auto cov21 = mfmat::covariance(mat2);
  auto mean2 = std::make_optional(mfmat::mean(mat2));
  auto cov22 = mfmat::covariance(mat2, mean2);
  BOOST_CHECK(cov21 == res2);
  BOOST_CHECK(cov22 == res2);
}


BOOST_AUTO_TEST_CASE(covariance_3x3)
{
  auto mat = mfmat::ct_mat<double, 3, 3>
    ({
      {  1.0,  3.0, -7.0 },
      {  3.0,  9.0,  2.0 },
      { -5.0,  4.0,  6.0 }
     });
  auto cov1 = mfmat::covariance(mat);
  auto mean = std::make_optional(mfmat::mean(mat));
  auto cov2 = mfmat::covariance(mat, mean);
  auto res = mfmat::ct_mat<double, 3, 3>
    ({
      {  11.5555555555, 5.1111111111, -10.2222222222 },
      {   5.1111111111, 6.8888888888,   5.2222222222 },
      { -10.2222222222, 5.2222222222,  29.5555555555 }
     });
  for (std::size_t i = 0; i < res.row_count; ++i)
    for (std::size_t j = 0; j < res.col_count; ++j)
    {
      auto cell1 = cov1[{i, j}];
      auto cell2 = cov2[{i, j}];
      auto cell3 = res[{i, j}];
      BOOST_CHECK_CLOSE(cell1, cell3, 0.0000001);
      BOOST_CHECK_CLOSE(cell2, cell2, 0.0000001);
    }
}


BOOST_AUTO_TEST_CASE(covariance_5x3)
{
  auto mat = mfmat::ct_mat<double, 5, 3>
    ({
      { 90.0, 80.0, 40.0 },
      { 90.0, 60.0, 80.0 },
      { 60.0, 50.0, 70.0 },
      { 30.0, 40.0, 70.0 },
      { 30.0, 20.0, 90.0 }
     });
  auto cov1 = mfmat::covariance(mat);
  auto mean = std::make_optional(mfmat::mean(mat));
  auto cov2 = mfmat::covariance(mat, mean);
  auto res = mfmat::ct_mat<double, 3, 3>
    ({
      {  720.0,  480.0, -240.0 },
      {  480.0,  400.0, -280.0 },
      { -240.0, -280.0,  280.0 }
     });
  BOOST_CHECK(cov1 == res);
  BOOST_CHECK(cov2 == res);
}


BOOST_AUTO_TEST_CASE(covariance_10x5)
{
  auto mat = mfmat::ct_mat<double, 10, 5>
    ({
      { -0.3,  4.2, -1.4,  7.3,  0.1 },
      {  3.2, -1.0,  3.0, -1.1,  2.5 },
      { -0.9,  0.2,  0.6,  0.5, -0.2 },
      { -1.5, -3.3, -5.5, -4.0, -2.5 },
      { -2.7, -0.2,  4.5,  2.5,  4.5 },
      { -0.1,  0.1,  0.2, -0.2, -0.2 },
      {  0.1,  8.4,  2.2,  3.2,  3.0 },
      {  1.0,  2.0,  1.0,  2.0,  1.0 },
      {  0.5,  1.5, -0.5,  3.4,  3.6 },
      { -2.5, -1.5, -2.7,  3.5, -1.3 }
    });
  auto cov1 = mfmat::covariance(mat);
  auto mean = std::make_optional(mfmat::mean(mat));
  auto cov2 = mfmat::covariance(mat, mean);
  auto res = mfmat::ct_mat<double, 5, 5>
    ({
      {  2.7176, 1.1508, 1.3728, -0.6148, 0.9480 },
      {  1.1508, 9.7864, 3.0834,  5.7486, 2.8840 },
      {  1.3728, 3.0834, 7.5044,  1.5486, 4.9040 },
      { -0.6148, 5.7486, 1.5486,  8.5849, 2.0505 },
      {  0.9480, 2.8840, 4.9040,  2.0505, 4.6465 }
    });
  for (std::size_t i = 0; i < res.row_count; ++i)
    for (std::size_t j = 0; j < res.col_count; ++j)
    {
      auto cell1 = cov1[{i, j}];
      auto cell2 = cov2[{i, j}];
      auto cell3 = res[{i, j}];
      BOOST_CHECK_CLOSE(cell1, cell3, 0.0000001);
      BOOST_CHECK_CLOSE(cell2, cell3, 0.0000001);
    }
}

BOOST_AUTO_TEST_SUITE_END()
