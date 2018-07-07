#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

namespace but = boost::unit_test;

BOOST_AUTO_TEST_SUITE(cl_matrices_ops_test_suite)

namespace
{

  void setup()
  {
    mfmat::cl_default_gpu_setter::instance();
    mfmat::cl_kernels_store::instance();
  }

} // !namespace anonymous

BOOST_AUTO_TEST_CASE(add_and_store, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{32, 32};
  mat_f += 1.1f;
  auto mat_f2 = mfmat::cl_mat_f{32, 32};
  mat_f2 += 2.2f;
  mat_f += mat_f2;
  for (auto i : mat_f)
    BOOST_CHECK(mfmat::are_equal(i, 3.3f));
  auto mat_d = mfmat::cl_mat_d{64, 64};
  mat_d += 3.3;
  auto mat_d2 = mfmat::cl_mat_d{64, 64};
  mat_d2 += 4.4;
  mat_d += mat_d2;
  for (auto i : mat_d)
    BOOST_CHECK(mfmat::are_equal(i, 7.7));
}


BOOST_AUTO_TEST_CASE(add, * but::fixture(&setup))
{
  auto mat_f1 = mfmat::cl_mat_f{32, 32};
  mat_f1 += 1.1f;
  auto mat_f2 = mfmat::cl_mat_f{32, 32};
  mat_f2 += 2.2f;
  auto mat_f = mat_f1 + mat_f2;
  for (auto i : mat_f)
    BOOST_CHECK(mfmat::are_equal(i, 3.3f));
  auto mat_d1 = mfmat::cl_mat_d{64, 64};
  mat_d1 += 3.3;
  auto mat_d2 = mfmat::cl_mat_d{64, 64};
  mat_d2 += 4.4;
  auto mat_d = mat_d1 + mat_d2;
  for (auto i : mat_d)
    BOOST_CHECK(mfmat::are_equal(i, 7.7));
}


BOOST_AUTO_TEST_CASE(self_add_and_store, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{50, 50};
  mat_f += 1.1f;
  mat_f += mat_f;
  for (auto i : mat_f)
    BOOST_CHECK(mfmat::are_equal(i, 2.2f));
  auto mat_d = mfmat::cl_mat_d{60, 60};
  mat_d += 3.3;
  mat_d += mat_d;
  for (auto i : mat_d)
    BOOST_CHECK(mfmat::are_equal(i, 6.6));
}


BOOST_AUTO_TEST_CASE(add_and_store_column, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{45, 50};
  auto mat_f2 = mfmat::cl_mat_f{45, 1};
  for (std::size_t i = 0; i < mat_f2.get_row_count(); ++i)
    mat_f2[{i, 0}] = 1.0f + static_cast<float>(i);
  mat_f += mat_f2;
  for (std::size_t m = 0; m < mat_f.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_f.get_col_count(); ++n)
    {
      auto cell = mat_f.get(m, n);
      auto cell2 = mat_f2[{m, 0}];
      BOOST_CHECK(mfmat::are_equal(cell, cell2));
    }
  auto mat_d = mfmat::cl_mat_d{50, 55};
  auto mat_d2 = mfmat::cl_mat_d{50, 1};
  for (std::size_t i = 0; i < mat_d2.get_row_count(); ++i)
    mat_d2[{i, 0}] = 1.0 + static_cast<double>(i);
  mat_d += mat_d2;
  for (std::size_t m = 0; m < mat_d.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_d.get_col_count(); ++n)
    {
      auto cell = mat_d.get(m, n);
      auto cell2 = mat_d2[{m, 0}];
      BOOST_CHECK(mfmat::are_equal(cell, cell2));
    }
}


BOOST_AUTO_TEST_CASE(add_column, * but::fixture(&setup))
{
  auto mat_f1 = mfmat::cl_mat_f{45, 50};
  auto mat_f2 = mfmat::cl_mat_f{45, 1};
  for (std::size_t i = 0; i < mat_f2.get_row_count(); ++i)
    mat_f2[{i, 0}] = 1.0f + static_cast<float>(i);
  auto mat_f = mat_f1 + mat_f2;
  for (std::size_t m = 0; m < mat_f.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_f.get_col_count(); ++n)
    {
      auto cell = mat_f.get(m, n);
      auto cell2 = mat_f2[{m, 0}];
      BOOST_CHECK(mfmat::are_equal(cell, cell2));
    }
  auto mat_d1 = mfmat::cl_mat_d{50, 55};
  auto mat_d2 = mfmat::cl_mat_d{50, 1};
  for (std::size_t i = 0; i < mat_d2.get_row_count(); ++i)
    mat_d2[{i, 0}] = 1.0 + static_cast<double>(i);
  auto mat_d = mat_d1 + mat_d2;
  for (std::size_t m = 0; m < mat_d.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_d.get_col_count(); ++n)
    {
      auto cell = mat_d.get(m, n);
      auto cell2 = mat_d2[{m, 0}];
      BOOST_CHECK(mfmat::are_equal(cell, cell2));
    }
}


BOOST_AUTO_TEST_CASE(add_and_store_row, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{55, 60};
  auto mat_f2 = mfmat::cl_mat_f{1, 60};
  for (std::size_t i = 0; i < mat_f2.get_col_count(); ++i)
    mat_f2[{0, i}] = 1.0f + static_cast<float>(i);
  mat_f += mat_f2;
  for (std::size_t m = 0; m < mat_f.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_f.get_col_count(); ++n)
    {
      auto cell = mat_f[{m, n}];
      auto cell2 = mat_f2.get(0, n);
      BOOST_CHECK(mfmat::are_equal(cell, cell2));
    }
  auto mat_d = mfmat::cl_mat_d{60, 65};
  auto mat_d2 = mfmat::cl_mat_d{1, 65};
  for (std::size_t i = 0; i < mat_d2.get_col_count(); ++i)
    mat_d2[{0, i}] = 1.0 + static_cast<double>(i);
  mat_d += mat_d2;
  for (std::size_t m = 0; m < mat_d.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_d.get_col_count(); ++n)
    {
      auto cell = mat_d[{m, n}];
      auto cell2 = mat_d2.get(0, n);
      BOOST_CHECK(mfmat::are_equal(cell, cell2));
    }
}


BOOST_AUTO_TEST_CASE(add_row, * but::fixture(&setup))
{
  auto mat_f1 = mfmat::cl_mat_f{55, 60};
  auto mat_f2 = mfmat::cl_mat_f{1, 60};
  for (std::size_t i = 0; i < mat_f2.get_col_count(); ++i)
    mat_f2[{0, i}] = 1.0f + static_cast<float>(i);
  auto mat_f = mat_f1 + mat_f2;
  for (std::size_t m = 0; m < mat_f.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_f.get_col_count(); ++n)
    {
      auto cell = mat_f[{m, n}];
      auto cell2 = mat_f2.get(0, n);
      BOOST_CHECK(mfmat::are_equal(cell, cell2));
    }
  auto mat_d1 = mfmat::cl_mat_d{60, 65};
  auto mat_d2 = mfmat::cl_mat_d{1, 65};
  for (std::size_t i = 0; i < mat_d2.get_col_count(); ++i)
    mat_d2[{0, i}] = 1.0 + static_cast<double>(i);
  auto mat_d = mat_d1 + mat_d2;
  for (std::size_t m = 0; m < mat_d.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_d.get_col_count(); ++n)
    {
      auto cell = mat_d[{m, n}];
      auto cell2 = mat_d2.get(0, n);
      BOOST_CHECK(mfmat::are_equal(cell, cell2));
    }
}


BOOST_AUTO_TEST_CASE(sub_and_store, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{40, 40};
  mat_f += 1.1f;
  auto mat_f2 = mfmat::cl_mat_f{40, 40};
  mat_f2 += 2.2f;
  mat_f -= mat_f2;
  for (auto i : mat_f)
    BOOST_CHECK(mfmat::are_equal(i, -1.1f));
  auto mat_d = mfmat::cl_mat_d{50, 50};
  mat_d += 3.3;
  auto mat_d2 = mfmat::cl_mat_d{50, 50};
  mat_d2 += 4.4;
  mat_d -= mat_d2;
  for (auto i : mat_d)
    BOOST_CHECK(mfmat::are_equal(i, -1.1));
}


BOOST_AUTO_TEST_CASE(sub, * but::fixture(&setup))
{
  auto mat_f1 = mfmat::cl_mat_f{40, 40};
  mat_f1 += 1.1f;
  auto mat_f2 = mfmat::cl_mat_f{40, 40};
  mat_f2 += 2.2f;
  auto mat_f = mat_f1 - mat_f2;
  for (auto i : mat_f)
    BOOST_CHECK(mfmat::are_equal(i, -1.1f));
  auto mat_d1 = mfmat::cl_mat_d{50, 50};
  mat_d1 += 3.3;
  auto mat_d2 = mfmat::cl_mat_d{50, 50};
  mat_d2 += 4.4;
  auto mat_d = mat_d1 - mat_d2;
  for (auto i : mat_d)
    BOOST_CHECK(mfmat::are_equal(i, -1.1));
}


BOOST_AUTO_TEST_CASE(self_sub_and_store, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{50, 50};
  mat_f += 1.1f;
  mat_f -= mat_f;
  for (auto i : mat_f)
    BOOST_CHECK(mfmat::are_equal(i, 0.0f));
  auto mat_d = mfmat::cl_mat_d{60, 60};
  mat_d += 3.3;
  mat_d -= mat_d;
  for (auto i : mat_d)
    BOOST_CHECK(mfmat::are_equal(i, 0.0));
}


BOOST_AUTO_TEST_CASE(sub_and_store_column, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{65, 70};
  auto mat_f2 = mfmat::cl_mat_f{65, 1};
  for (std::size_t i = 0; i < mat_f2.get_row_count(); ++i)
    mat_f2.get(i, 0) = 1.0f + static_cast<float>(i);
  mat_f -= mat_f2;
  for (std::size_t m = 0; m < mat_f.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_f.get_col_count(); ++n)
    {
      auto cell = mat_f[{m, n}];
      auto cell2 = mat_f2.get(m, 0);
      BOOST_CHECK(mfmat::are_equal(cell, -cell2));
    }
  auto mat_d = mfmat::cl_mat_d{75, 80};
  auto mat_d2 = mfmat::cl_mat_d{75, 1};
  for (std::size_t i = 0; i < mat_d2.get_row_count(); ++i)
    mat_d2.get(i, 0) = 1.0 + static_cast<double>(i);
  mat_d -= mat_d2;
  for (std::size_t m = 0; m < mat_d.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_d.get_col_count(); ++n)
    {
      auto cell = mat_d[{m, n}];
      auto cell2 = mat_d2.get(m, 0);
      BOOST_CHECK(mfmat::are_equal(cell, -cell2));
    }
}


BOOST_AUTO_TEST_CASE(sub_column, * but::fixture(&setup))
{
  auto mat_f1 = mfmat::cl_mat_f{65, 70};
  auto mat_f2 = mfmat::cl_mat_f{65, 1};
  for (std::size_t i = 0; i < mat_f2.get_row_count(); ++i)
    mat_f2.get(i, 0) = 1.0f + static_cast<float>(i);
  auto mat_f = mat_f1 - mat_f2;
  for (std::size_t m = 0; m < mat_f.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_f.get_col_count(); ++n)
    {
      auto cell = mat_f[{m, n}];
      auto cell2 = mat_f2.get(m, 0);
      BOOST_CHECK(mfmat::are_equal(cell, -cell2));
    }
  auto mat_d1 = mfmat::cl_mat_d{75, 80};
  auto mat_d2 = mfmat::cl_mat_d{75, 1};
  for (std::size_t i = 0; i < mat_d2.get_row_count(); ++i)
    mat_d2.get(i, 0) = 1.0 + static_cast<double>(i);
  auto mat_d = mat_d1 - mat_d2;
  for (std::size_t m = 0; m < mat_d.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_d.get_col_count(); ++n)
    {
      auto cell = mat_d[{m, n}];
      auto cell2 = mat_d2.get(m, 0);
      BOOST_CHECK(mfmat::are_equal(cell, -cell2));
    }
}


BOOST_AUTO_TEST_CASE(sub_and_store_row, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{75, 80};
  auto mat_f2 = mfmat::cl_mat_f{1, 80};
  for (std::size_t i = 0; i < mat_f2.get_col_count(); ++i)
    mat_f2.get(0, i) = 1.0f + static_cast<float>(i);
  mat_f -= mat_f2;
  for (std::size_t m = 0; m < mat_f.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_f.get_col_count(); ++n)
    {
      auto cell = mat_f.get(m, n);
      auto cell2 = mat_f2[{0, n}];
      BOOST_CHECK(mfmat::are_equal(cell, -cell2));
    }
  auto mat_d = mfmat::cl_mat_d{85, 90};
  auto mat_d2 = mfmat::cl_mat_d{1, 90};
  for (std::size_t i = 0; i < mat_d2.get_col_count(); ++i)
    mat_d2.get(0, i) = 1.0 + static_cast<double>(i);
  mat_d -= mat_d2;
  for (std::size_t m = 0; m < mat_d.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_d.get_col_count(); ++n)
    {
      auto cell = mat_d.get(m, n);
      auto cell2 = mat_d2[{0, n}];
      BOOST_CHECK(mfmat::are_equal(cell, -cell2));
    }
}


BOOST_AUTO_TEST_CASE(sub_row, * but::fixture(&setup))
{
  auto mat_f1 = mfmat::cl_mat_f{75, 80};
  auto mat_f2 = mfmat::cl_mat_f{1, 80};
  for (std::size_t i = 0; i < mat_f2.get_col_count(); ++i)
    mat_f2.get(0, i) = 1.0f + static_cast<float>(i);
  auto mat_f = mat_f1 - mat_f2;
  for (std::size_t m = 0; m < mat_f.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_f.get_col_count(); ++n)
    {
      auto cell = mat_f.get(m, n);
      auto cell2 = mat_f2[{0, n}];
      BOOST_CHECK(mfmat::are_equal(cell, -cell2));
    }
  auto mat_d1 = mfmat::cl_mat_d{85, 90};
  auto mat_d2 = mfmat::cl_mat_d{1, 90};
  for (std::size_t i = 0; i < mat_d2.get_col_count(); ++i)
    mat_d2.get(0, i) = 1.0 + static_cast<double>(i);
  auto mat_d = mat_d1 - mat_d2;
  for (std::size_t m = 0; m < mat_d.get_row_count(); ++m)
    for (std::size_t n = 0; n < mat_d.get_col_count(); ++n)
    {
      auto cell = mat_d.get(m, n);
      auto cell2 = mat_d2[{0, n}];
      BOOST_CHECK(mfmat::are_equal(cell, -cell2));
    }
}


BOOST_AUTO_TEST_CASE(incompatible_dimensions, * but::fixture(&setup))
{
  auto mat_f = mfmat::cl_mat_f{32, 32};
  auto mat_f2 = mfmat::cl_mat_f{32, 33};
  BOOST_CHECK_THROW(mat_f += mat_f2, std::out_of_range);
  BOOST_CHECK_THROW(mat_f + mat_f2, std::out_of_range);
  BOOST_CHECK_THROW(mat_f -= mat_f2, std::out_of_range);
  BOOST_CHECK_THROW(mat_f - mat_f2, std::out_of_range);
  auto mat_f3 = mfmat::cl_mat_f{1, 33};
  BOOST_CHECK_THROW(mat_f += mat_f3, std::out_of_range);
  BOOST_CHECK_THROW(mat_f + mat_f3, std::out_of_range);
  BOOST_CHECK_THROW(mat_f -= mat_f3, std::out_of_range);
  BOOST_CHECK_THROW(mat_f - mat_f3, std::out_of_range);
  auto mat_f4 = mfmat::cl_mat_f{33, 1};
  BOOST_CHECK_THROW(mat_f += mat_f4, std::out_of_range);
  BOOST_CHECK_THROW(mat_f + mat_f4, std::out_of_range);
  BOOST_CHECK_THROW(mat_f -= mat_f4, std::out_of_range);
  BOOST_CHECK_THROW(mat_f - mat_f4, std::out_of_range);
  auto mat_d = mfmat::cl_mat_d{64, 64};
  auto mat_d2 = mfmat::cl_mat_d{63, 64};
  BOOST_CHECK_THROW(mat_d += mat_d2, std::out_of_range);
  BOOST_CHECK_THROW(mat_d + mat_d2, std::out_of_range);
  BOOST_CHECK_THROW(mat_d -= mat_d2, std::out_of_range);
  BOOST_CHECK_THROW(mat_d - mat_d2, std::out_of_range);
  auto mat_d3 = mfmat::cl_mat_d{63, 1};
  BOOST_CHECK_THROW(mat_d += mat_d3, std::out_of_range);
  BOOST_CHECK_THROW(mat_d + mat_d3, std::out_of_range);
  BOOST_CHECK_THROW(mat_d -= mat_d3, std::out_of_range);
  BOOST_CHECK_THROW(mat_d - mat_d3, std::out_of_range);
  auto mat_d4 = mfmat::cl_mat_d{1, 63};
  BOOST_CHECK_THROW(mat_d += mat_d4, std::out_of_range);
  BOOST_CHECK_THROW(mat_d + mat_d4, std::out_of_range);
  BOOST_CHECK_THROW(mat_d -= mat_d4, std::out_of_range);
  BOOST_CHECK_THROW(mat_d - mat_d4, std::out_of_range);
}

BOOST_AUTO_TEST_SUITE_END()
