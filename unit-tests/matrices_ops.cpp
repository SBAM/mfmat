#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(matrices_ops_test_suite)

BOOST_AUTO_TEST_CASE(add_and_store)
{
  auto mat = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { 1, 1, 1 },
      { 1, 1, 1 }
     });
  auto mat2 = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { 2, 2, 2 },
      { 2, 2, 2 }
     });
  mat += mat2;
  auto mat3 = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { 3, 3, 3 },
      { 3, 3, 3 }
     });
  BOOST_CHECK(mat == mat3);
}


BOOST_AUTO_TEST_CASE(substract_and_store)
{
  auto mat = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { 1, 1, 1 },
      { 1, 1, 1 }
     });
  auto mat2 = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { 2, 2, 2 },
      { 2, 2, 2 }
     });
  mat -= mat2;
  auto mat3 = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { -1, -1, -1 },
      { -1, -1, -1 }
     });
  BOOST_CHECK(mat == mat3);
}


BOOST_AUTO_TEST_CASE(add)
{
  auto mat = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { 1, 1, 1 },
      { 1, 1, 1 }
     });
  auto mat2 = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { 2, 2, 2 },
      { 2, 2, 2 }
     });
  auto res = mat + mat2;
  auto mat3 = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { 3, 3, 3 },
      { 3, 3, 3 }
     });
  BOOST_CHECK(res == mat3);
}


BOOST_AUTO_TEST_CASE(add_double)
{
  auto mat = mfmat::dense_matrix<double, 2, 3>
    ({
      {  1.1,  1.2,  1.3 },
      { -1.1, -1.2, -1.3 }
     });
  auto mat2 = mfmat::dense_matrix<double, 2, 3>
    ({
      {  2.1,  2.2,  2.3 },
      { -2.1, -2.2, -2.3 }
     });
  auto res = mat + mat2;
  auto mat3 = mfmat::dense_matrix<double, 2, 3>
    ({
      {  3.2,  3.4,  3.6 },
      { -3.2, -3.4, -3.6 }
     });
  BOOST_CHECK(res == mat3);
}


BOOST_AUTO_TEST_CASE(substract)
{
  auto mat = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { 1, 1, 1 },
      { 1, 1, 1 }
     });
  auto mat2 = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { 2, 2, 2 },
      { 2, 2, 2 }
     });
  auto res = mat - mat2;
  auto mat3 = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { -1, -1, -1 },
      { -1, -1, -1 }
     });
  BOOST_CHECK(res == mat3);
}


BOOST_AUTO_TEST_CASE(substract_double)
{
  auto mat = mfmat::dense_matrix<double, 2, 3>
    ({
      {  1.1,  1.2,  1.3 },
      { -1.1, -1.2, -1.3 }
     });
  auto mat2 = mfmat::dense_matrix<double, 2, 3>
    ({
      {  2.1,  2.2,  2.3 },
      { -2.1, -2.2, -2.3 }
     });
  auto res = mat - mat2;
  auto mat3 = mfmat::dense_matrix<double, 2, 3>
    ({
      { -1.0, -1.0, -1.0 },
      {  1.0,  1.0,  1.0 }
     });
  BOOST_CHECK(res == mat3);
}


BOOST_AUTO_TEST_CASE(multiply_simple)
{
  auto mat = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { 1, 2, 3 },
      { 4, 5, 6 }
     });
  auto mat2 = mfmat::dense_matrix<std::int32_t, 3, 2>
    ({
      { 7, 8 },
      { 9, 10 },
      { 11, 12 }
     });
  auto mat3 = mfmat::dense_matrix<std::int32_t, 2, 2>
    ({
      { 58, 64 },
      { 139, 154 }
     });
  auto res = mat * mat2;
  BOOST_CHECK(res == mat3);
}


BOOST_AUTO_TEST_CASE(multiply_type_promotion)
{
  auto mat = mfmat::dense_matrix<std::int8_t, 1, 2>
    ({
      { 100, 100 }
     });
  auto mat2 = mfmat::dense_matrix<std::int32_t, 2, 1>
    ({
      { 100 },
      { 100 }
     });
  auto mat3 = mfmat::dense_matrix<std::int32_t, 1, 1>({ { 20000 } });
  auto res = mat * mat2;
  BOOST_CHECK(res == mat3);
  auto mat4 = mfmat::dense_matrix<std::int32_t, 1, 2>
    ({
      { 100, 100 }
     });
  auto mat5 = mfmat::dense_matrix<std::int8_t, 2, 1>
    ({
      { 100 },
      { 100 }
     });
  auto res2 = mat4 * mat5;
  BOOST_CHECK(res2 == mat3);
}


BOOST_AUTO_TEST_CASE(multiply_double)
{
  auto mat = mfmat::dense_matrix<double, 3, 4>
    ({
      {  1.2, -2.4,  2.4, -1.8 },
      {  4.2, -3.1,  0.9, -0.8 },
      {  1.1, -1.2,  2.2, -3.3 }
     });
  auto mat2 = mfmat::dense_matrix<double, 4, 4>
    ({
      {  1.0,  2.1, -0.8, -0.6 },
      {  0.5,  0.3,  1.5, -2.5 },
      {  0.1, -0.1,  2.5,  1.3 },
      {  2.3, -0.7, -0.5,  0.2 }
     });
  auto mat3 = mfmat::dense_matrix<double, 3, 4>
    ({
      { -3.90,  2.82,  2.34,  8.04 },
      {  0.90,  8.36, -5.36,  6.24 },
      { -6.87,  4.04,  4.47,  4.54 }
     });
  auto res = mat * mat2;
  BOOST_CHECK(res == mat3);
}


BOOST_AUTO_TEST_CASE(multiply_outer)
{
  auto mat = mfmat::dense_matrix<std::int32_t, 3, 2>
    ({
      {  1, 2 },
      {  3, 4 },
      {  5, 6 }
     });
  auto mat2 = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      {  1, 3, 5 },
      {  2, 4, 6 }
     });
  auto mat3 = mfmat::dense_matrix<std::int32_t, 3, 3>
    ({
      {  5, 11, 17 },
      { 11, 25, 39 },
      { 17, 39, 61 }
     });
  auto res = mat * mat2;
  BOOST_CHECK(res == mat3);
}

BOOST_AUTO_TEST_SUITE_END()
