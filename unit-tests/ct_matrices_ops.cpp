#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(matrices_ops_test_suite)

BOOST_AUTO_TEST_CASE(add_and_store)
{
  auto mat = mfmat::ct_mat<std::int32_t, 2, 3>
    ({
      { 1, 1, 1 },
      { 1, 1, 1 }
     });
  auto mat2 = mfmat::ct_mat<std::int32_t, 2, 3>
    ({
      { 2, 2, 2 },
      { 2, 2, 2 }
     });
  mat += mat2;
  auto mat3 = mfmat::ct_mat<std::int32_t, 2, 3>
    ({
      { 3, 3, 3 },
      { 3, 3, 3 }
     });
  BOOST_CHECK(mat == mat3);
}


BOOST_AUTO_TEST_CASE(substract_and_store)
{
  auto mat = mfmat::ct_mat<std::int32_t, 2, 3>
    ({
      { 1, 1, 1 },
      { 1, 1, 1 }
     });
  auto mat2 = mfmat::ct_mat<std::int32_t, 2, 3>
    ({
      { 2, 2, 2 },
      { 2, 2, 2 }
     });
  mat -= mat2;
  auto mat3 = mfmat::ct_mat<std::int32_t, 2, 3>
    ({
      { -1, -1, -1 },
      { -1, -1, -1 }
     });
  BOOST_CHECK(mat == mat3);
}


BOOST_AUTO_TEST_CASE(add)
{
  auto mat = mfmat::ct_mat<std::int32_t, 2, 3>
    ({
      { 1, 1, 1 },
      { 1, 1, 1 }
     });
  auto mat2 = mfmat::ct_mat<std::int32_t, 2, 3>
    ({
      { 2, 2, 2 },
      { 2, 2, 2 }
     });
  auto res = mat + mat2;
  auto mat3 = mfmat::ct_mat<std::int32_t, 2, 3>
    ({
      { 3, 3, 3 },
      { 3, 3, 3 }
     });
  BOOST_CHECK(res == mat3);
}


BOOST_AUTO_TEST_CASE(add_double)
{
  auto mat = mfmat::ct_mat<double, 2, 3>
    ({
      {  1.1,  1.2,  1.3 },
      { -1.1, -1.2, -1.3 }
     });
  auto mat2 = mfmat::ct_mat<double, 2, 3>
    ({
      {  2.1,  2.2,  2.3 },
      { -2.1, -2.2, -2.3 }
     });
  auto res = mat + mat2;
  auto mat3 = mfmat::ct_mat<double, 2, 3>
    ({
      {  3.2,  3.4,  3.6 },
      { -3.2, -3.4, -3.6 }
     });
  BOOST_CHECK(res == mat3);
}


BOOST_AUTO_TEST_CASE(substract)
{
  auto mat = mfmat::ct_mat<std::int32_t, 2, 3>
    ({
      { 1, 1, 1 },
      { 1, 1, 1 }
     });
  auto mat2 = mfmat::ct_mat<std::int32_t, 2, 3>
    ({
      { 2, 2, 2 },
      { 2, 2, 2 }
     });
  auto res = mat - mat2;
  auto mat3 = mfmat::ct_mat<std::int32_t, 2, 3>
    ({
      { -1, -1, -1 },
      { -1, -1, -1 }
     });
  BOOST_CHECK(res == mat3);
}


BOOST_AUTO_TEST_CASE(substract_double)
{
  auto mat = mfmat::ct_mat<double, 2, 3>
    ({
      {  1.1,  1.2,  1.3 },
      { -1.1, -1.2, -1.3 }
     });
  auto mat2 = mfmat::ct_mat<double, 2, 3>
    ({
      {  2.1,  2.2,  2.3 },
      { -2.1, -2.2, -2.3 }
     });
  auto res = mat - mat2;
  auto mat3 = mfmat::ct_mat<double, 2, 3>
    ({
      { -1.0, -1.0, -1.0 },
      {  1.0,  1.0,  1.0 }
     });
  BOOST_CHECK(res == mat3);
}

BOOST_AUTO_TEST_SUITE_END()
