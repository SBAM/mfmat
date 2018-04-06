#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(mean_test_suite)

BOOST_AUTO_TEST_CASE(identity_1x1_mean)
{
  auto mat1 = mfmat::ct_mat<std::int32_t, 1, 1>(mfmat::identity_tag());
  auto mean1 = mfmat::mean(mat1);
  BOOST_CHECK(mean1 == mat1);
  auto mat2 = mfmat::ct_mat<double, 1, 1>({{2.0}});
  auto mean2 = mfmat::mean(mat2);
  BOOST_CHECK(mean2 == mat2);
}


BOOST_AUTO_TEST_CASE(mean_3x3)
{
  auto mat = mfmat::ct_mat<double, 3, 3>
    ({
      { 1.0, 2.0, 3.0 },
      { 4.0, 5.0, 6.0 },
      { 7.0, 8.0, 9.0 }
     });
  auto mean = mfmat::mean(mat);
  auto res = mfmat::ct_mat<double, 1, 3>
    ({
      { 4.0, 5.0, 6.0 }
     });
  BOOST_CHECK(mean == res);
}


BOOST_AUTO_TEST_CASE(mean_5x3)
{
  auto mat = mfmat::ct_mat<double, 5, 3>
    ({
      { 90.0, 80.0, 40.0 },
      { 90.0, 60.0, 80.0 },
      { 60.0, 50.0, 70.0 },
      { 30.0, 40.0, 70.0 },
      { 30.0, 20.0, 90.0 }
     });
  auto mean = mfmat::mean(mat);
  auto res = mfmat::ct_mat<double, 1, 3>
    ({
      { 60.0, 50.0, 70.0 }
     });
  BOOST_CHECK(mean == res);
}

BOOST_AUTO_TEST_SUITE_END()
