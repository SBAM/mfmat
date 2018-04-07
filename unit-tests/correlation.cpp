#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(correlation_test_suite)

BOOST_AUTO_TEST_CASE(identity_2x2_correlation)
{
  auto mat = mfmat::ct_mat<double, 2, 2>(mfmat::identity_tag());
  auto res = mfmat::ct_mat<double, 2, 2>
    ({
      {  1.0, -1.0 },
      { -1.0,  1.0 }
     });
  auto cor1 = mfmat::correlation(mat);
  auto mean = std::make_optional(mfmat::mean(mat));
  auto cor2 = mfmat::correlation(mat, mean);
  auto stddev = std::make_optional(mfmat::std_dev(mat));
  auto cor3 = mfmat::correlation(mat, mean, stddev);
  //auto cor4 = mfmat::correlation(mat, std::nullopt, stddev);
  BOOST_CHECK(res == cor1);
  BOOST_CHECK(res == cor2);
  BOOST_CHECK(res == cor3);
  //BOOST_CHECK(res == cor4);
}


BOOST_AUTO_TEST_CASE(correlation_3x3)
{
  auto mat = mfmat::ct_mat<double, 3, 3>
    ({
      {  1.0,  3.0, -7.0 },
      {  3.0,  9.0,  2.0 },
      { -5.0,  4.0,  6.0 }
     });
  auto res = mfmat::ct_mat<double, 3, 3>
    ({
      {  1.0000, 0.5729, -0.5531 },
      {  0.5729, 1.0000,  0.3660 },
      { -0.5531, 0.3660,  1.0000 }
     });
  auto cor1 = mfmat::correlation(mat);
  auto mean = std::make_optional(mfmat::mean(mat));
  auto cor2 = mfmat::correlation(mat, mean);
  auto stddev = std::make_optional(mfmat::std_dev(mat));
  auto cor3 = mfmat::correlation(mat, mean, stddev);
  //auto cor4 = mfmat::correlation(mat, std::nullopt, stddev);
  for (std::size_t i = 0; i < res.row_count; ++i)
    for (std::size_t j = 0; j < res.col_count; ++j)
    {
      auto cell1 = cor1[{i, j}];
      auto cell2 = cor2[{i, j}];
      auto cell3 = cor3[{i, j}];
      //auto cell4 = cor4[{i, j}];
      auto cell5 = res[{i, j}];
      BOOST_CHECK_CLOSE(cell1, cell5, 0.01);
      BOOST_CHECK_CLOSE(cell2, cell5, 0.01);
      BOOST_CHECK_CLOSE(cell3, cell5, 0.01);
      //BOOST_CHECK_CLOSE(cell4, cell5, 0.01);
    }
}


BOOST_AUTO_TEST_CASE(correlation_5x3)
{
  auto mat = mfmat::ct_mat<double, 5, 3>
    ({
      { 90.0, 80.0, 40.0 },
      { 90.0, 60.0, 80.0 },
      { 60.0, 50.0, 70.0 },
      { 30.0, 40.0, 70.0 },
      { 30.0, 20.0, 90.0 }
     });
  auto res = mfmat::ct_mat<double, 3, 3>
    ({
      {  1.0000,  0.8944, -0.5345 },
      {  0.8944,  1.0000, -0.8367 },
      { -0.5345, -0.8367,  1.0000 }
     });
  auto cor1 = mfmat::correlation(mat);
  auto mean = std::make_optional(mfmat::mean(mat));
  auto cor2 = mfmat::correlation(mat, mean);
  auto stddev = std::make_optional(mfmat::std_dev(mat));
  auto cor3 = mfmat::correlation(mat, mean, stddev);
  //auto cor4 = mfmat::correlation(mat, std::nullopt, stddev);
  for (std::size_t i = 0; i < res.row_count; ++i)
    for (std::size_t j = 0; j < res.col_count; ++j)
    {
      auto cell1 = cor1[{i, j}];
      auto cell2 = cor2[{i, j}];
      auto cell3 = cor3[{i, j}];
      //auto cell4 = cor4[{i, j}];
      auto cell5 = res[{i, j}];
      BOOST_CHECK_CLOSE(cell1, cell5, 0.01);
      BOOST_CHECK_CLOSE(cell2, cell5, 0.01);
      BOOST_CHECK_CLOSE(cell3, cell5, 0.01);
      //BOOST_CHECK_CLOSE(cell4, cell5, 0.01);
    }
}


BOOST_AUTO_TEST_CASE(correlation_10x5)
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
  auto res = mfmat::ct_mat<double, 5, 5>
    ({
      {  1.0000, 0.2231, 0.3040, -0.1273, 0.2668 },
      {  0.2231, 1.0000, 0.3598,  0.6272, 0.4277 },
      {  0.3040, 0.3598, 1.0000,  0.1929, 0.8305 },
      { -0.1273, 0.6272, 0.1929,  1.0000, 0.3247 },
      {  0.2668, 0.4277, 0.8305,  0.3247, 1.0000 }
    });
  auto cor1 = mfmat::correlation(mat);
  auto mean = std::make_optional(mfmat::mean(mat));
  auto cor2 = mfmat::correlation(mat, mean);
  auto stddev = std::make_optional(mfmat::std_dev(mat));
  auto cor3 = mfmat::correlation(mat, mean, stddev);
  //auto cor4 = mfmat::correlation(mat, std::nullopt, stddev);
  for (std::size_t i = 0; i < res.row_count; ++i)
    for (std::size_t j = 0; j < res.col_count; ++j)
    {
      auto cell1 = cor1[{i, j}];
      auto cell2 = cor2[{i, j}];
      auto cell3 = cor3[{i, j}];
      //auto cell4 = cor4[{i, j}];
      auto cell5 = res[{i, j}];
      BOOST_CHECK_CLOSE(cell1, cell5, 0.1);
      BOOST_CHECK_CLOSE(cell2, cell5, 0.1);
      BOOST_CHECK_CLOSE(cell3, cell5, 0.1);
      //BOOST_CHECK_CLOSE(cell4, cell5, 0.1);
    }
}

BOOST_AUTO_TEST_SUITE_END()
