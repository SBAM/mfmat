#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

namespace but = boost::unit_test;

BOOST_AUTO_TEST_SUITE(cl_covariance_test_suite)

namespace
{

  void setup()
  {
    mfmat::cl_default_gpu_setter::instance();
    mfmat::cl_kernels_store::instance();
  }

} // !namespace anonymous

BOOST_AUTO_TEST_CASE(identity_1x1_covariance, * but::fixture(&setup))
{
  auto mat1_f = mfmat::cl_mat_f{1, mfmat::identity_tag{}};
  auto cov1_f = mfmat::covariance(mat1_f);
  BOOST_CHECK_EQUAL(cov1_f.get_row_count(), 1);
  BOOST_CHECK_EQUAL(cov1_f.get_col_count(), 1);
  BOOST_CHECK(mfmat::are_equal(cov1_f.get(0, 0), 0.0f));
  auto mat2_f = mfmat::cl_mat_f({{ 2.0f }});
  auto cov2_f = mfmat::covariance(mat2_f);
  BOOST_CHECK_EQUAL(cov2_f.get_row_count(), 1);
  BOOST_CHECK_EQUAL(cov2_f.get_col_count(), 1);
  BOOST_CHECK(mfmat::are_equal(cov2_f.get(0, 0), 0.0f));
  auto mat1_d = mfmat::cl_mat_d{1, mfmat::identity_tag{}};
  auto cov1_d = mfmat::covariance(mat1_d);
  BOOST_CHECK_EQUAL(cov1_d.get_row_count(), 1);
  BOOST_CHECK_EQUAL(cov1_d.get_col_count(), 1);
  BOOST_CHECK(mfmat::are_equal(cov1_d.get(0, 0), 0.0));
  auto mat2_d = mfmat::cl_mat_d({{ 2.0 }});
  auto cov2_d = mfmat::covariance(mat2_d);
  BOOST_CHECK_EQUAL(cov2_d.get_row_count(), 1);
  BOOST_CHECK_EQUAL(cov2_d.get_col_count(), 1);
  BOOST_CHECK(mfmat::are_equal(cov2_d.get(0, 0), 0.0));
}


BOOST_AUTO_TEST_CASE(covariance_3x3, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f
    ({
      {  1.0f,  3.0f, -7.0f },
      {  3.0f,  9.0f,  2.0f },
      { -5.0f,  4.0f,  6.0f }
     });
  auto res_f = mfmat::cl_mat_f
    ({
      {  11.5555555555f, 5.1111111111f, -10.2222222222f },
      {   5.1111111111f, 6.8888888888f,   5.2222222222f },
      { -10.2222222222f, 5.2222222222f,  29.5555555555f }
     });
  auto cov_f = mfmat::covariance(mat_f);
  BOOST_CHECK_EQUAL(cov_f.get_row_count(), 3);
  BOOST_CHECK_EQUAL(cov_f.get_col_count(), 3);
  for (std::size_t i = 0; i < res_f.get_row_count(); ++i)
    for (std::size_t j = 0; j < res_f.get_col_count(); ++j)
    {
      auto cell1 = res_f[{i, j}];
      auto cell2 = cov_f[{i, j}];
      BOOST_CHECK_CLOSE(cell1, cell2, 0.00001);
    }
  auto mat_d = mfmat::cl_mat_d
    ({
      {  1.0,  3.0, -7.0 },
      {  3.0,  9.0,  2.0 },
      { -5.0,  4.0,  6.0 }
     });
  auto res_d = mfmat::cl_mat_d
    ({
      {  11.5555555555, 5.1111111111, -10.2222222222 },
      {   5.1111111111, 6.8888888888,   5.2222222222 },
      { -10.2222222222, 5.2222222222,  29.5555555555 }
     });
  auto cov_d = mfmat::covariance(mat_d);
  BOOST_CHECK_EQUAL(cov_d.get_row_count(), 3);
  BOOST_CHECK_EQUAL(cov_d.get_col_count(), 3);
  for (std::size_t i = 0; i < res_d.get_row_count(); ++i)
    for (std::size_t j = 0; j < res_d.get_col_count(); ++j)
    {
      auto cell1 = res_d[{i, j}];
      auto cell2 = cov_d[{i, j}];
      BOOST_CHECK_CLOSE(cell1, cell2, 0.0000001);
    }
}


BOOST_AUTO_TEST_CASE(covariance_5x3, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f
    ({
      { 90.0f, 80.0f, 40.0f },
      { 90.0f, 60.0f, 80.0f },
      { 60.0f, 50.0f, 70.0f },
      { 30.0f, 40.0f, 70.0f },
      { 30.0f, 20.0f, 90.0f }
     });
  auto res_f = mfmat::cl_mat_f
    ({
      {  720.0f,  480.0f, -240.0f },
      {  480.0f,  400.0f, -280.0f },
      { -240.0f, -280.0f,  280.0f }
     });
  auto cov_f = mfmat::covariance(mat_f);
  BOOST_CHECK_EQUAL(cov_f.get_row_count(), 3);
  BOOST_CHECK_EQUAL(cov_f.get_col_count(), 3);
  for (std::size_t i = 0; i < res_f.get_row_count(); ++i)
    for (std::size_t j = 0; j < res_f.get_col_count(); ++j)
    {
      auto cell1 = res_f[{i, j}];
      auto cell2 = cov_f[{i, j}];
      BOOST_CHECK(mfmat::are_equal(cell1, cell2));
    }
  auto mat_d = mfmat::cl_mat_d
    ({
      { 90.0, 80.0, 40.0 },
      { 90.0, 60.0, 80.0 },
      { 60.0, 50.0, 70.0 },
      { 30.0, 40.0, 70.0 },
      { 30.0, 20.0, 90.0 }
     });
  auto res_d = mfmat::cl_mat_d
    ({
      {  720.0,  480.0, -240.0 },
      {  480.0,  400.0, -280.0 },
      { -240.0, -280.0,  280.0 }
     });
  auto cov_d = mfmat::covariance(mat_d);
  BOOST_CHECK_EQUAL(cov_d.get_row_count(), 3);
  BOOST_CHECK_EQUAL(cov_d.get_col_count(), 3);
  for (std::size_t i = 0; i < res_d.get_row_count(); ++i)
    for (std::size_t j = 0; j < res_d.get_col_count(); ++j)
    {
      auto cell1 = res_d[{i, j}];
      auto cell2 = cov_d[{i, j}];
      BOOST_CHECK(mfmat::are_equal(cell1, cell2));
    }
}


BOOST_AUTO_TEST_CASE(covariance_10x5, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f
    ({
      { -0.3f,  4.2f, -1.4f,  7.3f,  0.1f },
      {  3.2f, -1.0f,  3.0f, -1.1f,  2.5f },
      { -0.9f,  0.2f,  0.6f,  0.5f, -0.2f },
      { -1.5f, -3.3f, -5.5f, -4.0f, -2.5f },
      { -2.7f, -0.2f,  4.5f,  2.5f,  4.5f },
      { -0.1f,  0.1f,  0.2f, -0.2f, -0.2f },
      {  0.1f,  8.4f,  2.2f,  3.2f,  3.0f },
      {  1.0f,  2.0f,  1.0f,  2.0f,  1.0f },
      {  0.5f,  1.5f, -0.5f,  3.4f,  3.6f },
      { -2.5f, -1.5f, -2.7f,  3.5f, -1.3f }
    });
  auto res_f = mfmat::cl_mat_f
    ({
      {  2.7176f, 1.1508f, 1.3728f, -0.6148f, 0.9480f },
      {  1.1508f, 9.7864f, 3.0834f,  5.7486f, 2.8840f },
      {  1.3728f, 3.0834f, 7.5044f,  1.5486f, 4.9040f },
      { -0.6148f, 5.7486f, 1.5486f,  8.5849f, 2.0505f },
      {  0.9480f, 2.8840f, 4.9040f,  2.0505f, 4.6465f }
    });
  auto cov_f = mfmat::covariance(mat_f);
  BOOST_CHECK_EQUAL(cov_f.get_row_count(), 5);
  BOOST_CHECK_EQUAL(cov_f.get_col_count(), 5);
  for (std::size_t i = 0; i < res_f.get_row_count(); ++i)
    for (std::size_t j = 0; j < res_f.get_col_count(); ++j)
    {
      auto cell1 = res_f[{i, j}];
      auto cell2 = cov_f[{i, j}];
      BOOST_CHECK_CLOSE(cell1, cell2, 0.0001f);
    }
  auto mat_d = mfmat::cl_mat_d
    ({
      { -0.3,  4.2, -1.4,  7.3,  0.1 },
      {  3.2, -1.0,  3.0, -1.1,  2.5 },
      { -0.9,  0.2,  0.6,  0.5, -0.2 },
      { -1.5, -3.3, -5.5, -4.0, -2.5 },
      { -2.7, -0.2,  4.5,  2.5,  4.5 },
      { -0.1,  0.1,  0.2, -0.2, -0.2 },
      {  0.1,  8.4,  2.2,  3.2,  3.0 },
      {  1.0,  2.0,  1.0,  2.0,  1.0 },
      {  0.5,  1.5, -0.5,  3.4,  3.6 },
      { -2.5, -1.5, -2.7,  3.5, -1.3 }
    });
  auto res_d = mfmat::cl_mat_d
    ({
      {  2.7176, 1.1508, 1.3728, -0.6148, 0.9480 },
      {  1.1508, 9.7864, 3.0834,  5.7486, 2.8840 },
      {  1.3728, 3.0834, 7.5044,  1.5486, 4.9040 },
      { -0.6148, 5.7486, 1.5486,  8.5849, 2.0505 },
      {  0.9480, 2.8840, 4.9040,  2.0505, 4.6465 }
    });
  auto cov_d = mfmat::covariance(mat_d);
  BOOST_CHECK_EQUAL(cov_d.get_row_count(), 5);
  BOOST_CHECK_EQUAL(cov_d.get_col_count(), 5);
  for (std::size_t i = 0; i < res_d.get_row_count(); ++i)
    for (std::size_t j = 0; j < res_d.get_col_count(); ++j)
    {
      auto cell1 = res_d[{i, j}];
      auto cell2 = cov_d[{i, j}];
      BOOST_CHECK_CLOSE(cell1, cell2, 0.0000001);
    }
}


BOOST_AUTO_TEST_CASE(covariance_empty_matrix, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{0, 0};
  BOOST_CHECK_THROW(mfmat::covariance(mat_f), std::runtime_error);
  auto mat_d = mfmat::cl_mat_d{0, 0};
  BOOST_CHECK_THROW(mfmat::covariance(mat_d), std::runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
