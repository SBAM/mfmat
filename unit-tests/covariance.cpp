#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(covariance_test_suite)

constexpr auto owr = mfmat::op_way::row;
constexpr auto owc = mfmat::op_way::col;

BOOST_AUTO_TEST_CASE(identity_1x1_covariance)
{
  auto mat1 = mfmat::ct_mat<std::int32_t, 1, 1>(mfmat::identity_tag());
  auto res1 = mfmat::ct_mat<std::int32_t, 1, 1>();
  auto cov1_1 = mfmat::covariance<owr>(mat1);
  BOOST_CHECK(cov1_1 == res1);
  auto cov1_2 = mfmat::covariance<owc>(mat1);
  BOOST_CHECK(cov1_2 == res1);
  auto mat2 = mfmat::ct_mat<double, 1, 1>({{2.0}});
  auto res2 = mfmat::ct_mat<double, 1, 1>();
  auto cov2_1 = mfmat::covariance<owr>(mat2);
  BOOST_CHECK(cov2_1 == res2);
  auto cov2_2 = mfmat::covariance<owc>(mat2);
  BOOST_CHECK(cov2_2 == res2);
}


BOOST_AUTO_TEST_CASE(covariance_3x3)
{
  auto mat = mfmat::ct_mat<double, 3, 3>
    ({
      {  1.0,  3.0, -7.0 },
      {  3.0,  9.0,  2.0 },
      { -5.0,  4.0,  6.0 }
     });
  auto cov_col = mfmat::covariance<owc>(mat);
  auto res_col = mfmat::ct_mat<double, 3, 3>
    ({
      {  11.5555555555, 5.1111111111, -10.2222222222 },
      {   5.1111111111, 6.8888888888,   5.2222222222 },
      { -10.2222222222, 5.2222222222,  29.5555555555 }
     });
  for (std::size_t i = 0; i < res_col.row_count; ++i)
    for (std::size_t j = 0; j < res_col.col_count; ++j)
    {
      auto cell1 = cov_col[{i, j}];
      auto cell2 = res_col[{i, j}];
      BOOST_CHECK_CLOSE(cell1, cell2, 0.0000001);
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
  auto cov_col = mfmat::covariance<owc>(mat);
  auto res_col = mfmat::ct_mat<double, 3, 3>
    ({
      {  720.0,  480.0, -240.0 },
      {  480.0,  400.0, -280.0 },
      { -240.0, -280.0,  280.0 }
     });
  BOOST_CHECK(cov_col == res_col);
}

BOOST_AUTO_TEST_SUITE_END()
