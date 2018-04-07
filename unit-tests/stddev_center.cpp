#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(stddev_center_test_suite)

BOOST_AUTO_TEST_CASE(identity_2x2_stddev_center)
{
  auto mat = mfmat::ct_mat<double, 2, 2>(mfmat::identity_tag());
  auto res = mfmat::ct_mat<double, 2, 2>
    ({
      {  1.0, -1.0 },
      { -1.0,  1.0 }
     });
  auto sc1 = mat;
  sc1.stddev_center();
  auto mean = std::make_optional(mfmat::mean(mat));
  auto sc2 = mat;
  sc2.stddev_center(mean);
  auto stddev = std::make_optional(mfmat::std_dev(mat));
  auto sc3 = mat;
  sc3.stddev_center(mean, stddev);
  auto sc4 = mat;
  sc4.stddev_center(std::nullopt, stddev);
  BOOST_CHECK(res == sc1);
  BOOST_CHECK(res == sc2);
  BOOST_CHECK(res == sc3);
  BOOST_CHECK(res == sc4);
}


BOOST_AUTO_TEST_CASE(stddev_center_3x3)
{
  auto mat = mfmat::ct_mat<double, 3, 3>
    ({
      { 1.0, 2.0, 3.0 },
      { 4.0, 5.0, 6.0 },
      { 7.0, 8.0, 9.0 }
     });
  auto sc1 = mat;
  sc1.stddev_center();
  auto mean = std::make_optional(mfmat::mean(mat));
  auto sc2 = mat;
  sc2.stddev_center(mean);
  auto stddev = std::make_optional(mfmat::std_dev(mat));
  auto sc3 = mat;
  sc3.stddev_center(mean, stddev);
  auto sc4 = mat;
  sc4.stddev_center(std::nullopt, stddev);
  auto res = mfmat::ct_mat<double, 3, 3>
    ({
      { -1.2247, -1.2247, -1.2247 },
      {  0.0,     0.0,     0.0 },
      {  1.2247,  1.2247,  1.2247 }
     });
  for (std::size_t i = 0; i < res.row_count; ++i)
    for (std::size_t j = 0; j < res.col_count; ++j)
    {
      auto cell1 = sc1[{i, j}];
      auto cell2 = sc2[{i, j}];
      auto cell3 = sc3[{i, j}];
      auto cell4 = sc4[{i, j}];
      auto cell5 = res[{i, j}];
      BOOST_CHECK_CLOSE(cell1, cell5, 0.01);
      BOOST_CHECK_CLOSE(cell2, cell5, 0.01);
      BOOST_CHECK_CLOSE(cell3, cell5, 0.01);
      BOOST_CHECK_CLOSE(cell4, cell5, 0.01);
    }
}


BOOST_AUTO_TEST_CASE(stddev_center_5x3)
{
  auto mat = mfmat::ct_mat<double, 5, 3>
    ({
      { 90.0, 80.0, 40.0 },
      { 90.0, 60.0, 80.0 },
      { 60.0, 50.0, 70.0 },
      { 30.0, 40.0, 70.0 },
      { 30.0, 20.0, 90.0 }
     });
  auto sc1 = mat;
  sc1.stddev_center();
  auto mean = std::make_optional(mfmat::mean(mat));
  auto sc2 = mat;
  sc2.stddev_center(mean);
  auto stddev = std::make_optional(mfmat::std_dev(mat));
  auto sc3 = mat;
  sc3.stddev_center(mean, stddev);
  auto sc4 = mat;
  sc4.stddev_center(std::nullopt, stddev);
  auto res = mfmat::ct_mat<double, 5, 3>
    ({
      {  1.118,  1.5, -1.7928 },
      {  1.118,  0.5,  0.5976 },
      {  0.0  ,  0.0,  0.0    },
      { -1.118, -0.5,  0.0    },
      { -1.118, -1.5,  1.1952 }
     });
  for (std::size_t i = 0; i < res.row_count; ++i)
    for (std::size_t j = 0; j < res.col_count; ++j)
    {
      auto cell1 = sc1[{i, j}];
      auto cell2 = sc2[{i, j}];
      auto cell3 = sc3[{i, j}];
      auto cell4 = sc4[{i, j}];
      auto cell5 = res[{i, j}];
      BOOST_CHECK_CLOSE(cell1, cell5, 0.01);
      BOOST_CHECK_CLOSE(cell2, cell5, 0.01);
      BOOST_CHECK_CLOSE(cell3, cell5, 0.01);
      BOOST_CHECK_CLOSE(cell4, cell5, 0.01);
    }
}

BOOST_AUTO_TEST_SUITE_END()
