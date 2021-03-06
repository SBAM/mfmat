#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/tools/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(ct_transpose_test_suite)

BOOST_AUTO_TEST_CASE(transpose_1x1)
{
  auto mat = mfmat::ct_mat<std::int32_t, 1, 1>(mfmat::identity_tag{});
  auto res = transpose(mat);
  BOOST_CHECK(mat == res);
  res.transpose();
  BOOST_CHECK(mat == res);
}


BOOST_AUTO_TEST_CASE(transpose_3x3)
{
  auto mat = mfmat::ct_mat<std::int32_t, 3, 3>
    ({
      { -2,  2, -3 },
      { -1,  1,  3 },
      {  2,  0, -1 }
     });
  auto res = transpose(mat);
  auto mat_res = mfmat::ct_mat<std::int32_t, 3, 3>
    ({
      { -2, -1,  2 },
      {  2,  1,  0 },
      { -3,  3, -1 }
     });
  BOOST_CHECK(res == mat_res);
  mat.transpose();
  BOOST_CHECK(mat == mat_res);
}


BOOST_AUTO_TEST_CASE(transpose_2x4)
{
  auto mat = mfmat::ct_mat<std::int32_t, 2, 4>
    ({
      { 1, 2, 3, 4 },
      { 5, 6, 7, 8 }
     });
  auto res = transpose(mat);
  auto mat_res = mfmat::ct_mat<std::int32_t, 4, 2>
    ({
      { 1, 5 },
      { 2, 6 },
      { 3, 7 },
      { 4, 8 }
     });
  BOOST_CHECK(res == mat_res);
}


BOOST_AUTO_TEST_CASE(transpose_4x2_double)
{
  auto mat = mfmat::ct_mat<double, 4, 2>
    ({
      { 1.1, 5.5 },
      { 2.2, 6.6 },
      { 3.3, 7.7 },
      { 4.4, 8.8 }
    });
  auto res = transpose(mat);
  auto mat_res = mfmat::ct_mat<double, 2, 4>
    ({
      { 1.1, 2.2, 3.3, 4.4 },
      { 5.5, 6.6, 7.7, 8.8 }
    });
  BOOST_CHECK(res == mat_res);
}

BOOST_AUTO_TEST_SUITE_END()
