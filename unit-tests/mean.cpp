#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(mean_test_suite)

constexpr auto owr = mfmat::op_way::row;
constexpr auto owc = mfmat::op_way::col;

BOOST_AUTO_TEST_CASE(identity_1x1_mean)
{
  auto mat1 = mfmat::ct_mat<std::int32_t, 1, 1>(mfmat::identity_tag());
  auto mean1_1 = mfmat::mean<owr>(mat1).get<0, 0>();
  BOOST_CHECK(mean1_1 == 1);
  auto mean1_2 = mfmat::mean<owc>(mat1).get<0, 0>();
  BOOST_CHECK(mean1_2 == 1);
  auto mat2 = mfmat::ct_mat<double, 1, 1>({{2.0}});
  auto mean2_1 = mfmat::mean<owr>(mat2).get<0, 0>();
  BOOST_CHECK_CLOSE(mean2_1, 2.0, 0.001);
  auto mean2_2 = mfmat::mean<owc>(mat2).get<0, 0>();
  BOOST_CHECK_CLOSE(mean2_2, 2.0, 0.001);
}


BOOST_AUTO_TEST_CASE(mean_3x3)
{
  auto mat = mfmat::ct_mat<double, 3, 3>
    ({
      { 1.0, 2.0, 3.0 },
      { 4.0, 5.0, 6.0 },
      { 7.0, 8.0, 9.0 }
     });
  auto mean_row = mfmat::mean<owr>(mat);
  auto res_row = mfmat::ct_mat<double, 3, 1>
    ({
      { 2.0 },
      { 5.0 },
      { 8.0 }
     });
  BOOST_CHECK(mean_row == res_row);
  auto mean_col = mfmat::mean<owc>(mat);
  auto res_col = mfmat::ct_mat<double, 1, 3>
    ({
      { 4.0, 5.0, 6.0 }
     });
  BOOST_CHECK(mean_col == res_col);
}


BOOST_AUTO_TEST_CASE(mean_5x3)
{
  auto mat = mfmat::ct_mat<double, 5, 3>
    ({
      { 90.0, 80.0, 40.0 },
      { 90.0, 60.0, 80.0 },
      { 60.0, 50.0, 70.0 },
      { 30.0, 40.0, 70.0 },
      { 30.0, 20.0, 90.0 },
     });
  auto mean_col = mfmat::mean<owc>(mat);
  auto res_col = mfmat::ct_mat<double, 1, 3>
    ({
      { 60.0, 50.0, 70.0 }
     });
  BOOST_CHECK(mean_col == res_col);
  auto mean_row = mfmat::mean<owr>(mat);
  auto res_row = mfmat::ct_mat<double, 5, 1>
    ({
      { 210.0 / 3.0 },
      { 230.0 / 3.0 },
      { 180.0 / 3.0 },
      { 140.0 / 3.0 },
      { 140.0 / 3.0 }
     });
  BOOST_CHECK(mean_col == res_col);
}

BOOST_AUTO_TEST_SUITE_END()
