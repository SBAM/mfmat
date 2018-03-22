#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(orthonormalization_test_suite)

BOOST_AUTO_TEST_CASE(identity_1x1)
{
  auto mat = mfmat::ct_mat<std::int32_t, 1, 1>(mfmat::identity_tag());
  auto on = orthonormalize(mat);
  BOOST_CHECK(mat == on);
}


BOOST_AUTO_TEST_CASE(identity_5x5)
{
  auto orig = mfmat::ct_mat<double, 5, 5>(mfmat::identity_tag());
  auto mat = orig * 5.0;
  auto on = orthonormalize(mat);
  BOOST_CHECK(on == orig);
}


BOOST_AUTO_TEST_CASE(upper_triangular_3x3)
{
  auto orig = mfmat::ct_mat<double, 3, 3>(mfmat::identity_tag());
  auto mat = mfmat::ct_mat<double, 3, 3>
    ({
      { 3, 1, 1 },
      { 0, 3, 1 },
      { 0, 0, 3 }
     });
  auto on = orthonormalize(mat);
  BOOST_CHECK(on == orig);
}


BOOST_AUTO_TEST_CASE(test_1_3x3)
{
  auto mat = mfmat::ct_mat<double, 3, 3>
    ({
      { 3, 1, 1 },
      { 1, 3, 1 },
      { 1, 1, 3 }
     });
  auto res = mfmat::ct_mat<double, 3, 3>
    ({
      {   3.0 * std::sqrt(11)   / 11.0 ,
        - 5.0 * std::sqrt(22.0) / 66.0 ,
        - 1.0 * std::sqrt(2.0)  /  6.0 },
      {   1.0 * std::sqrt(11)   / 11.0 ,
         13.0 * std::sqrt(22.0) / 66.0 ,
        - 1.0 * std::sqrt(2.0)  /  6.0 },
      {   1.0 * std::sqrt(11)   / 11.0 ,
          1.0 * std::sqrt(22.0) / 33.0 ,
          2.0 * std::sqrt(2.0)  /  3.0 }
     });
  auto on = orthonormalize(mat);
  BOOST_CHECK(on == res);
}

BOOST_AUTO_TEST_SUITE_END()
