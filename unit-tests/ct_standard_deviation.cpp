#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(ct_standard_deviation_test_suite)

BOOST_AUTO_TEST_CASE(identity_1x1_standard_deviation)
{
  auto mat1 = mfmat::ct_mat<std::int32_t, 1, 1>(mfmat::identity_tag());
  auto sd1 = mfmat::std_dev(mat1);
  auto res1 = mfmat::ct_mat<std::int32_t, 1, 1>();
  BOOST_CHECK(sd1 == res1);
  auto mat2 = mfmat::ct_mat<double, 1, 1>({{2.0}});
  auto sd2 = mfmat::std_dev(mat2);
  auto res2 = mfmat::ct_mat<double, 1, 1>();
  BOOST_CHECK(sd2 == res2);
}


BOOST_AUTO_TEST_CASE(standard_deviation_3x3)
{
  auto mat = mfmat::ct_mat<double, 3, 3>
    ({
      { 1.0, 2.0, 3.0 },
      { 4.0, 5.0, 6.0 },
      { 7.0, 8.0, 9.0 }
     });
  auto sd1 = mfmat::std_dev(mat);
  auto mean = std::make_optional(mfmat::mean(mat));
  auto sd2 = mfmat::std_dev(mat, mean);
  auto res = mfmat::ct_mat<double, 1, 3>
    ({
      { 2.4495, 2.4495, 2.4495 }
     });
  for (std::size_t i = 0; i < res.row_count; ++i)
    for (std::size_t j = 0; j < res.col_count; ++j)
    {
      auto cell1 = sd1[{i, j}];
      auto cell2 = sd2[{i, j}];
      auto cell3 = res[{i, j}];
      BOOST_CHECK_CLOSE(cell1, cell3, 0.001);
      BOOST_CHECK_CLOSE(cell2, cell3, 0.001);
    }
}


BOOST_AUTO_TEST_CASE(standard_deviation_5x3)
{
  auto mat = mfmat::ct_mat<double, 5, 3>
    ({
      { 90.0, 80.0, 40.0 },
      { 90.0, 60.0, 80.0 },
      { 60.0, 50.0, 70.0 },
      { 30.0, 40.0, 70.0 },
      { 30.0, 20.0, 90.0 }
     });
  auto sd1 = mfmat::std_dev(mat);
  auto mean = std::make_optional(mfmat::mean(mat));
  auto sd2 = mfmat::std_dev(mat, mean);
  auto res = mfmat::ct_mat<double, 1, 3>
    ({
      { 26.8328, 20.0, 16.7332 }
     });
  for (std::size_t i = 0; i < res.row_count; ++i)
    for (std::size_t j = 0; j < res.col_count; ++j)
    {
      auto cell1 = sd1[{i, j}];
      auto cell2 = sd2[{i, j}];
      auto cell3 = res[{i, j}];
      BOOST_CHECK_CLOSE(cell1, cell3, 0.001);
      BOOST_CHECK_CLOSE(cell2, cell3, 0.001);
    }
}

BOOST_AUTO_TEST_SUITE_END()
