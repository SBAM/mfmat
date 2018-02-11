#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(compare_ops_test_suite)

BOOST_AUTO_TEST_CASE(integer_compare)
{
  auto mat = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { 1, 1, 1 },
      { 1, 1, 1 }
     });
  auto mat2 = mat;
  BOOST_CHECK(mat == mat2);
}


BOOST_AUTO_TEST_CASE(integer_diff)
{
  auto mat = mfmat::dense_matrix<std::int32_t, 1, 1>({{ 0 }});
  auto mat2 = mfmat::dense_matrix<std::int32_t, 1, 1>({{ 1 }});
  BOOST_CHECK(mat != mat2);
}


BOOST_AUTO_TEST_CASE(double_compare)
{
  auto mat = mfmat::dense_matrix<double, 2, 3>
    ({
      {  99999999999999999.0,  1.0,  0.000000000000000001 },
      { -99999999999999999.0, -1.0, -0.000000000000000001 }
     });
  auto mat2 = mat;
  BOOST_CHECK(mat == mat2);
}


BOOST_AUTO_TEST_CASE(double_diff1)
{
  auto mat = mfmat::dense_matrix<double, 1, 1>({{  99999999999999999.0 }});
  auto mat2 = mfmat::dense_matrix<double, 1, 1>({{ 99999999999999955.0 }});
  BOOST_CHECK(mat != mat2);
}


BOOST_AUTO_TEST_CASE(double_diff2)
{
  auto mat = mfmat::dense_matrix<double, 1, 1>({{  -99999999999999999.0 }});
  auto mat2 = mfmat::dense_matrix<double, 1, 1>({{ -99999999999999955.0 }});
  BOOST_CHECK(mat != mat2);
}


BOOST_AUTO_TEST_CASE(double_diff3)
{
  auto mat = mfmat::dense_matrix<double, 1, 1>({{  0.000000000000000001 }});
  auto mat2 = mfmat::dense_matrix<double, 1, 1>({{ 0.000000000000000002 }});
  BOOST_CHECK(mat != mat2);
}


BOOST_AUTO_TEST_CASE(double_diff4)
{
  auto mat = mfmat::dense_matrix<double, 1, 1>({{  -0.000000000000000001 }});
  auto mat2 = mfmat::dense_matrix<double, 1, 1>({{ -0.000000000000000002 }});
  BOOST_CHECK(mat != mat2);
}


BOOST_AUTO_TEST_SUITE_END()
