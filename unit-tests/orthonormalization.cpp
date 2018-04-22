#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(orthonormalization_test_suite)

template <typename T>
void check_tolerance(const T& lhs, const T& rhs, double tolerance)
{
  for (std::size_t r = 0; r < lhs.row_count; ++r)
    for (std::size_t c = 0; c < lhs.col_count; ++c)
    {
      auto cell_1 = lhs[{r, c}];
      auto cell_2 = rhs[{r, c}];
      BOOST_CHECK_SMALL(cell_1 - cell_2, tolerance);
    }
}


template <typename T>
void check_transpose_is_inverse(const T& mat, double tolerance = 0.0)
{
  static_assert(mat.row_count == mat.col_count);
  auto identity = T(mfmat::identity_tag{});
  auto mul_by_transpose = mat * transpose(mat);
  if (mfmat::is_zero(tolerance))
  {
    BOOST_CHECK(identity == mul_by_transpose);
  }
  else
    check_tolerance(identity, mul_by_transpose, tolerance);
}


BOOST_AUTO_TEST_CASE(identity_1x1)
{
  auto mat = mfmat::ct_mat<double, 1, 1>(mfmat::identity_tag());
  auto on = mat;
  mat.orthonormalize();
  BOOST_CHECK(mat == on);
  check_transpose_is_inverse(on);
}


BOOST_AUTO_TEST_CASE(identity_5x5)
{
  auto orig = mfmat::ct_mat<double, 5, 5>(mfmat::identity_tag());
  auto mat = orig * 5.0;
  auto on = mat;
  on.orthonormalize();
  BOOST_CHECK(on == orig);
  check_transpose_is_inverse(on);
}


BOOST_AUTO_TEST_CASE(upper_triangular_3x3)
{
  auto orig = mfmat::ct_mat<double, 3, 3>(mfmat::identity_tag());
  auto mat = mfmat::ct_mat<double, 3, 3>
    ({
      { 3.0, 1.0, 1.0 },
      { 0.0, 3.0, 1.0 },
      { 0.0, 0.0, 3.0 }
     });
  auto on = mat;
  on.orthonormalize();
  BOOST_CHECK(on == orig);
  check_transpose_is_inverse(on);
}


BOOST_AUTO_TEST_CASE(test_1_3x3)
{
  auto mat = mfmat::ct_mat<double, 3, 3>
    ({
      { 3.0, 1.0, 1.0 },
      { 1.0, 3.0, 1.0 },
      { 1.0, 1.0, 3.0 }
     });
  auto res = mfmat::ct_mat<double, 3, 3>
    ({
      {   3.0 * std::sqrt(11.0) / 11.0 ,
        - 5.0 * std::sqrt(22.0) / 66.0 ,
        - 1.0 * std::sqrt(2.0)  /  6.0 },
      {   1.0 * std::sqrt(11.0) / 11.0 ,
         13.0 * std::sqrt(22.0) / 66.0 ,
        - 1.0 * std::sqrt(2.0)  /  6.0 },
      {   1.0 * std::sqrt(11.0) / 11.0 ,
          1.0 * std::sqrt(22.0) / 33.0 ,
          2.0 * std::sqrt(2.0)  /  3.0 }
     });
  auto on = mat;
  on.orthonormalize();
  BOOST_CHECK(on == res);
  check_transpose_is_inverse(on);
}


BOOST_AUTO_TEST_CASE(test_2_3x3)
{
  auto mat = mfmat::ct_mat<double, 3, 3>
    ({
      { 1.0, 1.0, 1.0 },
      { 2.0, 1.0, 0.0 },
      { 5.0, 1.0, 3.0 }
     });
  auto res = mfmat::ct_mat<double, 3, 3>
    ({
      {   1.0 * std::sqrt(30.0)  /  30.0 ,
         11.0 * std::sqrt(195.0) / 195.0 ,
          3.0 * std::sqrt(26.0)  /  26.0  },
      {   1.0 * std::sqrt(30.0)  /  15.0 ,
          7.0 * std::sqrt(195.0) / 195.0 ,
        - 2.0 * std::sqrt(26.0)  /  13.0  },
      {   1.0 * std::sqrt(30.0)  /   6.0 ,
        - 1.0 * std::sqrt(195.0) /  39.0 ,
          1.0 * std::sqrt(26.0)  /  26.0  }
     });
  auto on = mat;
  on.orthonormalize();
  BOOST_CHECK(on == res);
  check_transpose_is_inverse(on);
}


BOOST_AUTO_TEST_CASE(test_3_3x3)
{
  auto mat = mfmat::ct_mat<double, 3, 3>
    ({
      { 1.0, 1.0, 1.0 },
      { 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0 }
     });
  auto res = mfmat::ct_mat<double, 3, 3>
    ({
      { 1.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0 }
     });
  auto on = mat;
  on.orthonormalize();
  BOOST_CHECK(on == res);
}


BOOST_AUTO_TEST_CASE(test_4_3x3)
{
  auto mat = mfmat::ct_mat<double, 3, 3>
    ({
      {  1.0,  3.0,  5.0 },
      {  7.0, 11.0, 13.0 },
      { 17.0, 19.0, 23.0 }
     });
  auto res = mfmat::ct_mat<double, 3, 3>
    ({
      {    1.0 * std::sqrt(   339.0) /    339.0 ,
         307.0 * std::sqrt(342390.0) / 342390.0 ,
          27.0 * std::sqrt(  1010.0) /   1010.0 },
      {    7.0 * std::sqrt(   339.0) /    339.0 ,
         227.0 * std::sqrt(342390.0) / 171195.0 ,
        -  8.0 * std::sqrt(  1010.0) /    505.0 },
      {   17.0 * std::sqrt(   339.0) /    339.0 ,
        - 41.0 * std::sqrt(342390.0) /  68478.0 ,
           1.0 * std::sqrt(  1010.0) /   202.0 }
     });
  auto on = mat;
  on.orthonormalize();
  check_tolerance(on, res, 1.0e-14);
  check_transpose_is_inverse(on, 1.0e-14);
}


BOOST_AUTO_TEST_CASE(test_5x3)
{
  auto mat = mfmat::ct_mat<double, 5, 3>
    ({
      {  1.0,  0.0, -1.0 },
      { -1.0,  1.0,  1.0 },
      {  0.0,  0.0,  1.0 },
      {  1.0,  2.0, -2.0 },
      {  0.0, -1.0,  2.0 }
     });
  auto res = mfmat::ct_mat<double, 5, 3>
    ({
      {   1.0 * std::sqrt(  3.0) /   3.0 ,
        - 1.0 * std::sqrt( 51.0) /  51.0 ,
          1.0 * std::sqrt(238.0) / 238.0 },
      { - 1.0 * std::sqrt(  3.0) /   3.0 ,
          4.0 * std::sqrt( 51.0) /  51.0 ,
          9.0 * std::sqrt(238.0) / 476.0 },
      {   0.0                            ,
          0.0                            ,
          1.0 * std::sqrt(238.0) /  28.0 },
      {   1.0 * std::sqrt(  3.0) /   3.0 ,
          5.0 * std::sqrt( 51.0) /  51.0 ,
          1.0 * std::sqrt(238.0) /  68.0 },
      {   0.0                            ,
        - 1.0 * std::sqrt( 51.0) /  17.0 ,
         23.0 * std::sqrt(238.0) / 476.0 }
     });
  auto on = mat;
  on.orthonormalize();
  check_tolerance(on, res, 1.0e-14);
}


BOOST_AUTO_TEST_CASE(test_5x5)
{
  auto mat = mfmat::ct_mat<double, 5, 5>
    ({
      {  1.0,  0.0, -1.0, -2.0,  2.0 },
      { -1.0,  1.0,  1.0,  0.0, -2.0 },
      {  0.0,  2.0,  1.0, -1.0,  0.0 },
      {  1.0,  2.0, -2.0, -1.0,  1.0 },
      {  0.0, -1.0,  2.0,  1.0, -2.0 }
     });
  auto res = mfmat::ct_mat<double, 5, 5>
    ({
      {   1.0 * std::sqrt(   3.0) /    3.0 ,
        - 1.0 * std::sqrt(  87.0) /   87.0 ,
          4.0 * std::sqrt(1131.0) / 1131.0 ,
        -30.0 * std::sqrt(  13.0) /  143.0 ,
        - 5.0 * std::sqrt(   3.0) /   33.0 },
      { - 1.0 * std::sqrt(   3.0) /    3.0 ,
          4.0 * std::sqrt(  87.0) /   87.0 ,
        - 1.0 * std::sqrt(1131.0) /  754.0 ,
        -23.0 * std::sqrt(  13.0) /  286.0 ,
        - 4.0 * std::sqrt(   3.0) /   11.0 },
      {   0.0                              ,
          2.0 * std::sqrt(  87.0) /   29.0 ,
          1.0 * std::sqrt(1131.0) /   58.0 ,
        - 1.0 * std::sqrt(  13.0) /  22.0 ,
          3.0 * std::sqrt(   3.0) /   11.0 },
      {   1.0 * std::sqrt(   3.0) /    3.0 ,
          5.0 * std::sqrt(  87.0) /   87.0 ,
        -11.0 * std::sqrt(1131.0) / 2262.0 ,
         37.0 * std::sqrt(  13.0) /  286.0 ,
        - 7.0 * std::sqrt(   3.0) /   33.0 },
      {   0.0                              ,
        - 1.0 * std::sqrt(  87.0) /   29.0 ,
         53.0 * std::sqrt(1131.0) / 2262.0 ,
         25.0 * std::sqrt(  13.0) /  286.0 ,
        - 8.0 * std::sqrt(   3.0) /   33.0 }
     });
  auto on = mat;
  on.orthonormalize();
  BOOST_CHECK(on == res);
  check_transpose_is_inverse(on);
}


BOOST_AUTO_TEST_CASE(test_3x5_1)
{
  auto mat = mfmat::ct_mat<double, 3, 5>
    ({
      { 1.0, 0.0, 0.0, 1.0, 3.0 },
      { 0.0, 1.0, 0.0, 2.0, 2.0 },
      { 0.0, 0.0, 1.0, 3.0, 1.0 }
     });
  auto res = mfmat::ct_mat<double, 3, 5>
    ({
      { 1.0, 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 1.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 1.0, 0.0, 0.0 }
     });
  auto on = mat;
  on.orthonormalize();
  BOOST_CHECK(on == res);
}


BOOST_AUTO_TEST_CASE(test_3x5_2)
{
  auto mat = mfmat::ct_mat<double, 3, 5>
    ({
      { -1.0, 9.0,  2.0, 8.0,  7.0 },
      {  5.0, 6.0, -5.0, 7.0,  2.0 },
      { -9.0, 0.0,  1.0, 2.0, -3.0 }
     });
  auto res = mfmat::ct_mat<double, 3, 5>
    ({
      {  -1.0 * std::sqrt(107.0)    / 107.0,
        164.0 * std::sqrt(143594.0) / 71797.0,
          9.0 * std::sqrt(1342.0)   / 671.0,    0.0, 0.0 },
      {   5.0 * std::sqrt(107.0)    / 107.0,
        179.0 * std::sqrt(143594.0) / 143594.0,
        -27.0 * std::sqrt(1342.0)   / 1342.0,   0.0, 0.0 },
      {  -9.0 * std::sqrt(107.0)    / 107.0,
         63.0 * std::sqrt(143594.0) / 143594.0,
        -17.0 * std::sqrt(1342.0)   / 1342.0,   0.0, 0.0 }
     });
  auto on = mat;
  on.orthonormalize();
  BOOST_CHECK(on == res);
}

BOOST_AUTO_TEST_SUITE_END()
