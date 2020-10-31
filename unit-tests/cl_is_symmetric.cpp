#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/tools/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

namespace but = boost::unit_test;

BOOST_AUTO_TEST_SUITE(cl_is_symmetric_test_suite)

namespace
{

  void setup()
  {
    mfmat::cl_default_gpu_setter::instance();
    mfmat::cl_kernels_store::instance();
  }

} // !namespace anonymous

BOOST_AUTO_TEST_CASE(is_symmetric_1, * but::fixture(&setup))
{
  auto mat1 = mfmat::cl_mat_f(1);
  BOOST_CHECK(mat1.is_symmetric());
  auto mat2 = mfmat::cl_mat_d(1, mfmat::identity_tag{});
  BOOST_CHECK(mat2.is_symmetric());
  auto mat3 = mfmat::cl_mat_d(2);
  BOOST_CHECK(mat3.is_symmetric());
  auto mat4 = mfmat::cl_mat_f(2, mfmat::identity_tag{});
  BOOST_CHECK(mat4.is_symmetric());
  mat4 *= 2.0f;
  BOOST_CHECK(mat4.is_symmetric());
  auto mat5 = mfmat::cl_mat_d
    ({
      { 1.0, 0.0, 0.0 },
      { 0.0, 2.0, 0.0 },
      { 0.0, 0.0, 3.0 }
     });
  BOOST_CHECK(mat5.is_symmetric());
}


BOOST_AUTO_TEST_CASE(is_symmetric_2, * but::fixture(&setup))
{
  auto mat1 = mfmat::cl_mat_f
    ({
      { 1.0f, 1.0f },
      { 0.0f, 1.0f }
     });
  BOOST_CHECK(!mat1.is_symmetric());
  auto mat2 = mfmat::cl_mat_d
    ({
      { 1.0, 0.0, 0.0 },
      { 0.0, 2.0, 0.0 },
      { 1.0, 0.0, 3.0 }
     });
  BOOST_CHECK(!mat2.is_symmetric());
}


BOOST_AUTO_TEST_CASE(is_symmetric_3, * but::fixture(&setup))
{
  auto mat1 = mfmat::cl_mat_f(200, mfmat::identity_tag{});
  BOOST_CHECK(mat1.is_symmetric());
  mat1.get(50, 100) = 10.0f;
  BOOST_CHECK(!mat1.is_symmetric());
  mat1.get(100, 50) = 10.0f;
  BOOST_CHECK(mat1.is_symmetric());
  auto mat2 = mfmat::cl_mat_d(300, mfmat::identity_tag{});
  BOOST_CHECK(mat2.is_symmetric());
  mat2.get(150, 200) = 20.0;
  BOOST_CHECK(!mat2.is_symmetric());
  mat2.get(200, 150) = 20.0;
  BOOST_CHECK(mat2.is_symmetric());
}

BOOST_AUTO_TEST_SUITE_END()
