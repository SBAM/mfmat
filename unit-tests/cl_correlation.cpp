#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/tools/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

namespace but = boost::unit_test;

BOOST_AUTO_TEST_SUITE(cl_correlation_test_suite)

namespace
{

  void setup()
  {
    mfmat::cl_default_gpu_setter::instance();
    mfmat::cl_kernels_store::instance();
  }

} // !namespace anonymous

BOOST_AUTO_TEST_CASE(identity_2x2_correlation, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{2, mfmat::identity_tag{}};
  auto cor_f = mfmat::correlation(mat_f);
  BOOST_CHECK_EQUAL(cor_f.get_row_count(), 2);
  BOOST_CHECK_EQUAL(cor_f.get_col_count(), 2);
  for (std::size_t m = 0; m < cor_f.get_row_count(); ++m)
    for (std::size_t n = 0; n < cor_f.get_col_count(); ++n)
      if (m == n)
        BOOST_CHECK(mfmat::are_equal(cor_f.get(m, n), 1.0f));
      else
        BOOST_CHECK(mfmat::are_equal(cor_f.get(m, n), -1.0f));
  auto mat_d = mfmat::cl_mat_d{2, mfmat::identity_tag{}};
  auto cor_d = mfmat::correlation(mat_d);
  BOOST_CHECK_EQUAL(cor_d.get_row_count(), 2);
  BOOST_CHECK_EQUAL(cor_d.get_col_count(), 2);
  for (std::size_t m = 0; m < cor_d.get_row_count(); ++m)
    for (std::size_t n = 0; n < cor_d.get_col_count(); ++n)
      if (m == n)
        BOOST_CHECK(mfmat::are_equal(cor_d.get(m, n), 1.0));
      else
        BOOST_CHECK(mfmat::are_equal(cor_d.get(m, n), -1.0));
}


BOOST_AUTO_TEST_CASE(correlation_3x3, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f
    ({
      {  1.0f,  3.0f, -7.0f },
      {  3.0f,  9.0f,  2.0f },
      { -5.0f,  4.0f,  6.0f }
    });
  auto res_f = mfmat::cl_mat_f
    ({
      {  1.0000f, 0.5729f, -0.5531f },
      {  0.5729f, 1.0000f,  0.3660f },
      { -0.5531f, 0.3660f,  1.0000f }
    });
  auto cor_f = mfmat::correlation(mat_f);
  BOOST_CHECK_EQUAL(cor_f.get_row_count(), 3);
  BOOST_CHECK_EQUAL(cor_f.get_col_count(), 3);
  for (std::size_t i = 0; i < res_f.get_row_count(); ++i)
    for (std::size_t j = 0; j < res_f.get_col_count(); ++j)
    {
      auto cell1 = res_f[{i, j}];
      auto cell2 = cor_f[{i, j}];
      BOOST_CHECK_CLOSE(cell1, cell2, 0.01);
    }
  auto mat_d = mfmat::cl_mat_d
    ({
      {  1.0,  3.0, -7.0 },
      {  3.0,  9.0,  2.0 },
      { -5.0,  4.0,  6.0 }
    });
  auto res_d = mfmat::cl_mat_d
    ({
      {  1.0000, 0.5729, -0.5531 },
      {  0.5729, 1.0000,  0.3660 },
      { -0.5531, 0.3660,  1.0000 }
    });
  auto cor_d = mfmat::correlation(mat_d);
  BOOST_CHECK_EQUAL(cor_d.get_row_count(), 3);
  BOOST_CHECK_EQUAL(cor_d.get_col_count(), 3);
  for (std::size_t i = 0; i < res_d.get_row_count(); ++i)
    for (std::size_t j = 0; j < res_d.get_col_count(); ++j)
    {
      auto cell1 = res_d[{i, j}];
      auto cell2 = cor_d[{i, j}];
      BOOST_CHECK_CLOSE(cell1, cell2, 0.01);
    }
}


BOOST_AUTO_TEST_CASE(correlation_5x3, * but::fixture(&setup))
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
      {  1.0000f,  0.8944f, -0.5345f },
      {  0.8944f,  1.0000f, -0.8367f },
      { -0.5345f, -0.8367f,  1.0000f }
     });
  auto cor_f = mfmat::correlation(mat_f);
  BOOST_CHECK_EQUAL(cor_f.get_row_count(), 3);
  BOOST_CHECK_EQUAL(cor_f.get_col_count(), 3);
  for (std::size_t i = 0; i < res_f.get_row_count(); ++i)
    for (std::size_t j = 0; j < res_f.get_col_count(); ++j)
    {
      auto cell1 = res_f[{i, j}];
      auto cell2 = cor_f[{i, j}];
      BOOST_CHECK_CLOSE(cell1, cell2, 0.01);
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
      {  1.0000,  0.8944, -0.5345 },
      {  0.8944,  1.0000, -0.8367 },
      { -0.5345, -0.8367,  1.0000 }
     });
  auto cor_d = mfmat::correlation(mat_d);
  BOOST_CHECK_EQUAL(cor_d.get_row_count(), 3);
  BOOST_CHECK_EQUAL(cor_d.get_col_count(), 3);
  for (std::size_t i = 0; i < res_d.get_row_count(); ++i)
    for (std::size_t j = 0; j < res_d.get_col_count(); ++j)
    {
      auto cell1 = res_d[{i, j}];
      auto cell2 = cor_d[{i, j}];
      BOOST_CHECK_CLOSE(cell1, cell2, 0.01);
    }
}


BOOST_AUTO_TEST_CASE(correlation_10x5, * but::fixture(&setup))
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
      {  1.0000f, 0.2231f, 0.3040f, -0.1273f, 0.2668f },
      {  0.2231f, 1.0000f, 0.3598f,  0.6272f, 0.4277f },
      {  0.3040f, 0.3598f, 1.0000f,  0.1929f, 0.8305f },
      { -0.1273f, 0.6272f, 0.1929f,  1.0000f, 0.3247f },
      {  0.2668f, 0.4277f, 0.8305f,  0.3247f, 1.0000f }
    });
  auto cor_f = mfmat::correlation(mat_f);
  BOOST_CHECK_EQUAL(cor_f.get_row_count(), 5);
  BOOST_CHECK_EQUAL(cor_f.get_col_count(), 5);
  for (std::size_t i = 0; i < res_f.get_row_count(); ++i)
    for (std::size_t j = 0; j < res_f.get_col_count(); ++j)
    {
      auto cell1 = res_f[{i, j}];
      auto cell2 = cor_f[{i, j}];
      BOOST_CHECK_CLOSE(cell1, cell2, 0.1);
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
      {  1.0000, 0.2231, 0.3040, -0.1273, 0.2668 },
      {  0.2231, 1.0000, 0.3598,  0.6272, 0.4277 },
      {  0.3040, 0.3598, 1.0000,  0.1929, 0.8305 },
      { -0.1273, 0.6272, 0.1929,  1.0000, 0.3247 },
      {  0.2668, 0.4277, 0.8305,  0.3247, 1.0000 }
    });
  auto cor_d = mfmat::correlation(mat_d);
  BOOST_CHECK_EQUAL(cor_d.get_row_count(), 5);
  BOOST_CHECK_EQUAL(cor_d.get_col_count(), 5);
  for (std::size_t i = 0; i < res_d.get_row_count(); ++i)
    for (std::size_t j = 0; j < res_d.get_col_count(); ++j)
    {
      auto cell1 = res_d[{i, j}];
      auto cell2 = cor_d[{i, j}];
      BOOST_CHECK_CLOSE(cell1, cell2, 0.1);
    }
}


BOOST_AUTO_TEST_CASE(correlation_empty_matrix, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{0, 0};
  BOOST_CHECK_THROW(mfmat::correlation(mat_f), std::runtime_error);
  auto mat_d = mfmat::cl_mat_d{0, 0};
  BOOST_CHECK_THROW(mfmat::correlation(mat_d), std::runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
