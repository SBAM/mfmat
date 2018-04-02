#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(deviation_test_suite)

constexpr auto owr = mfmat::op_way::row;
constexpr auto owc = mfmat::op_way::col;

BOOST_AUTO_TEST_CASE(identity_1x1_deviation)
{
  auto mat1 = mfmat::ct_mat<std::int32_t, 1, 1>(mfmat::identity_tag());
  auto res1 = mfmat::ct_mat<std::int32_t, 1, 1>();
  auto dev1_1 = mfmat::deviation<owr>(mat1);
  BOOST_CHECK(dev1_1 == res1);
  auto dev1_2 = mfmat::deviation<owc>(mat1);
  BOOST_CHECK(dev1_2 == res1);
  auto mat2 = mfmat::ct_mat<double, 1, 1>({{2.0}});
  auto res2 = mfmat::ct_mat<double, 1, 1>();
  auto dev2_1 = mfmat::deviation<owr>(mat2);
  BOOST_CHECK(dev2_1 == res2);
  auto dev2_2 = mfmat::deviation<owc>(mat2);
  BOOST_CHECK(dev2_2 == res2);
}


BOOST_AUTO_TEST_CASE(deviation_3x3)
{
  auto mat = mfmat::ct_mat<double, 3, 3>
    ({
      { 1.0, 2.0, 3.0 },
      { 4.0, 5.0, 6.0 },
      { 7.0, 8.0, 9.0 }
     });
  auto dev_row = mfmat::deviation<owr>(mat);
  auto res_row = mfmat::ct_mat<double, 3, 3>
    ({
      { -1.0,  0.0,  1.0 },
      { -1.0,  0.0,  1.0 },
      { -1.0,  0.0,  1.0 }
     });
  BOOST_CHECK(dev_row == res_row);
  auto dev_col = mfmat::deviation<owc>(mat);
  auto res_col = mfmat::ct_mat<double, 3, 3>
    ({
      { -3.0, -3.0, -3.0 },
      {  0.0,  0.0,  0.0 },
      {  3.0,  3.0,  3.0 }
     });
  BOOST_CHECK(dev_col == res_col);
}


BOOST_AUTO_TEST_CASE(deviation_5x3)
{
  auto mat = mfmat::ct_mat<double, 5, 3>
    ({
      { 90.0, 80.0, 40.0 },
      { 90.0, 60.0, 80.0 },
      { 60.0, 50.0, 70.0 },
      { 30.0, 40.0, 70.0 },
      { 30.0, 20.0, 90.0 }
     });
  auto dev_col = mfmat::deviation<owc>(mat);
  auto res_col = mfmat::ct_mat<double, 5, 3>
    ({
      {  30.0,  30.0, -30.0 },
      {  30.0,  10.0,  10.0 },
      {   0.0,   0.0,   0.0 },
      { -30.0, -10.0,   0.0 },
      { -30.0, -30.0,  20.0 }
     });
  BOOST_CHECK(dev_col == res_col);
  auto dev_row = mfmat::deviation<owr>(mat);
  auto res_row = mfmat::ct_mat<double, 5, 3>
    ({
      { 90.0 - 210.0 / 3.0,
        80.0 - 210.0 / 3.0,
        40.0 - 210.0 / 3.0 },
      { 90.0 - 230.0 / 3.0,
        60.0 - 230.0 / 3.0,
        80.0 - 230.0 / 3.0 },
      { 60.0 - 180.0 / 3.0,
        50.0 - 180.0 / 3.0,
        70.0 - 180.0 / 3.0 },
      { 30.0 - 140.0 / 3.0,
        40.0 - 140.0 / 3.0,
        70.0 - 140.0 / 3.0 },
      { 30.0 - 140.0 / 3.0,
        20.0 - 140.0 / 3.0,
        90.0 - 140.0 / 3.0 }
     });
  BOOST_CHECK(dev_col == res_col);
}

BOOST_AUTO_TEST_SUITE_END()
