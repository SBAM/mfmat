#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(is_symmetric_test_suite)

BOOST_AUTO_TEST_CASE(is_symmetric_1)
{
  auto mat1 = mfmat::ct_mat<std::int32_t, 1, 1>();
  BOOST_CHECK(mat1.is_symmetric());
  auto mat2 = mfmat::ct_mat<std::int32_t, 1, 1>(mfmat::identity_tag{});
  BOOST_CHECK(mat2.is_symmetric());
  auto mat3 = mfmat::ct_mat<std::int32_t, 2, 2>();
  BOOST_CHECK(mat3.is_symmetric());
  auto mat4 = mfmat::ct_mat<std::int32_t, 2, 2>(mfmat::identity_tag{});
  BOOST_CHECK(mat4.is_symmetric());
  mat4 *= 2;
  BOOST_CHECK(mat4.is_symmetric());
  auto mat5 = mfmat::ct_mat<std::int32_t, 3, 3>
    ({
      { 1, 0, 0 },
      { 0, 2, 0 },
      { 0, 0, 3 }
     });
  BOOST_CHECK(mat5.is_symmetric());
}


BOOST_AUTO_TEST_CASE(is_symmetric_2)
{
  auto mat1 = mfmat::ct_mat<std::int32_t, 2, 2>
    ({
      { 1, 1 },
      { 0, 1 }
     });
  BOOST_CHECK(!mat1.is_symmetric());
  auto mat2 = mfmat::ct_mat<std::int32_t, 3, 3>
    ({
      { 1, 0, 0 },
      { 0, 2, 0 },
      { 1, 0, 3 }
     });
  BOOST_CHECK(!mat2.is_symmetric());
}


BOOST_AUTO_TEST_CASE(is_symmetric_3)
{
  auto mat1 = mfmat::ct_mat<double, 1, 1>();
  BOOST_CHECK(mat1.is_symmetric());
  auto mat2 = mfmat::ct_mat<double, 1, 1>(mfmat::identity_tag{});
  BOOST_CHECK(mat2.is_symmetric());
  auto mat3 = mfmat::ct_mat<double, 2, 2>();
  BOOST_CHECK(mat3.is_symmetric());
  auto mat4 = mfmat::ct_mat<double, 2, 2>(mfmat::identity_tag{});
  BOOST_CHECK(mat4.is_symmetric());
  mat4 *= 2.0;
  BOOST_CHECK(mat4.is_symmetric());
  auto mat5 = mfmat::ct_mat<double, 3, 3>
    ({
      { 1.0, 0.0, 0.0 },
      { 0.0, 2.0, 0.0 },
      { 0.0, 0.0, 3.0 }
     });
  BOOST_CHECK(mat5.is_symmetric());
}


BOOST_AUTO_TEST_CASE(is_symmetric_4)
{
  auto mat1 = mfmat::ct_mat<double, 2, 2>
    ({
      { 1.0, 1.0 },
      { 0.0, 1.0 }
     });
  BOOST_CHECK(!mat1.is_symmetric());
  auto mat2 = mfmat::ct_mat<double, 3, 3>
    ({
      { 1.0, 0.0, 0.0 },
      { 0.0, 2.0, 0.0 },
      { 1.0, 0.0, 3.0 }
     });
  BOOST_CHECK(!mat2.is_symmetric());
}


BOOST_AUTO_TEST_CASE(is_symmetric_5)
{
  static constexpr auto eps = std::numeric_limits<double>::epsilon();
  auto mat1 = mfmat::ct_mat<double, 3, 3>
    ({
      { 1.0, eps, eps },
      { eps, 2.0, eps },
      { eps, eps, 3.0 }
     });
  BOOST_CHECK(mat1.is_symmetric());
  auto mat2 = mat1 * 2.0;
  BOOST_CHECK(mat2.is_symmetric());
  static constexpr auto eps2 = eps * 2.1;
  auto mat3 = mfmat::ct_mat<double, 3, 3>
    ({
      { 1.0,  eps,  eps },
      { eps2, 2.0,  eps },
      { eps2, eps2, 3.0 }
     });
  BOOST_CHECK(!mat3.is_symmetric());
}

BOOST_AUTO_TEST_SUITE_END()
