#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/tools/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

namespace but = boost::unit_test;

BOOST_AUTO_TEST_SUITE(cl_is_diagonal_test_suite)

namespace
{

  void setup()
  {
    mfmat::cl_default_gpu_setter::instance();
    mfmat::cl_kernels_store::instance();
  }

} // !namespace anonymous

BOOST_AUTO_TEST_CASE(is_diagonal_1, * but::fixture(&setup))
{
  auto mat1 = mfmat::cl_mat_f(1);
  BOOST_CHECK(mat1.is_diagonal());
  auto mat2 = mfmat::cl_mat_d(1, mfmat::identity_tag{});
  BOOST_CHECK(mat2.is_diagonal());
  auto mat3 = mfmat::cl_mat_d(2);
  BOOST_CHECK(mat3.is_diagonal());
  auto mat4 = mfmat::cl_mat_f(2, mfmat::identity_tag{});
  BOOST_CHECK(mat4.is_diagonal());
  mat4 *= 2.0f;
  BOOST_CHECK(mat4.is_diagonal());
  auto mat5 = mfmat::cl_mat_f
    ({
      { 1.0f, 0.0f, 0.0f },
      { 0.0f, 2.0f, 0.0f },
      { 0.0f, 0.0f, 3.0f }
     });
  BOOST_CHECK(mat5.is_diagonal());
  auto mat6 = mfmat::cl_mat_d
    ({
      { 1.0, 0.0, 0.0, 0.0 },
      { 0.0, 2.0, 0.0, 0.0 },
      { 0.0, 0.0, 3.0, 0.0 }
     });
  BOOST_CHECK(mat6.is_diagonal());
  auto mat7 = mfmat::cl_mat_f
    ({
      { 1.0f, 0.0f, 0.0f },
      { 0.0f, 2.0f, 0.0f },
      { 0.0f, 0.0f, 3.0f },
      { 0.0f, 0.0f, 0.0f }
     });
  BOOST_CHECK(mat7.is_diagonal());
}


BOOST_AUTO_TEST_CASE(is_diagonal_2, * but::fixture(&setup))
{
  auto mat1 = mfmat::cl_mat_f
    ({
      { 1.0f, 1.0f },
      { 0.0f, 1.0f }
     });
  BOOST_CHECK(!mat1.is_diagonal());
  auto mat2 = mfmat::cl_mat_d
    ({
      { 1.0, 0.0, 0.0 },
      { 0.0, 2.0, 0.0 },
      { 1.0, 0.0, 3.0 }
     });
  BOOST_CHECK(!mat2.is_diagonal());
  auto mat3 = mfmat::cl_mat_f
    ({
      { 1.0f, 0.0f, 0.0f, 1.0f },
      { 0.0f, 2.0f, 0.0f, 0.0f },
      { 0.0f, 0.0f, 3.0f, 0.0f }
     });
  BOOST_CHECK(!mat3.is_diagonal());
  auto mat4 = mfmat::cl_mat_d
    ({
      { 1.0, 0.0, 0.0 },
      { 0.0, 2.0, 0.0 },
      { 0.0, 0.0, 3.0 },
      { 1.0, 0.0, 0.0 }
     });
  BOOST_CHECK(!mat4.is_diagonal());
}


BOOST_AUTO_TEST_CASE(is_diagonal_3, * but::fixture(&setup))
{
  auto mat1 = mfmat::cl_mat_f(300, mfmat::identity_tag{});
  mat1 *= 2.0f;
  BOOST_CHECK(mat1.is_diagonal());
  mat1.get(150, 149) = 0.00001f;
  BOOST_CHECK(!mat1.is_diagonal());
  auto mat2 = mfmat::cl_mat_d(400, mfmat::identity_tag{});
  mat2 *= 1.5;
  BOOST_CHECK(mat2.is_diagonal());
  mat2.get(250, 249) = 0.00001;
  BOOST_CHECK(!mat2.is_diagonal());
}


BOOST_AUTO_TEST_CASE(is_diagonal_4, * but::fixture(&setup))
{
  auto mat1 = mfmat::cl_mat_f(300, 350);
  mat1.get(100, 100) = 2.0f;
  BOOST_CHECK(mat1.is_diagonal());
  mat1.get(250, 249) = 0.00001f;
  BOOST_CHECK(!mat1.is_diagonal());
  auto mat2 = mfmat::cl_mat_d(400, 450);
  mat2.get(200, 200) = 1.5;
  BOOST_CHECK(mat2.is_diagonal());
  mat2.get(350, 349) = 0.00001;
  BOOST_CHECK(!mat2.is_diagonal());
}

BOOST_AUTO_TEST_SUITE_END()
