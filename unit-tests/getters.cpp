#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(getters_test_suite)

BOOST_AUTO_TEST_CASE(runtime_getter)
{
  auto mat = mfmat::dense_matrix<std::int32_t, 5, 10>();
  for (std::size_t i = 0; i < mat.row_count; ++i)
    for (std::size_t j = 0; j < mat.col_count; ++j)
    {
      auto cell = mat[{i, j}];
      BOOST_CHECK(cell == 0);
    }
}


BOOST_AUTO_TEST_CASE(compile_time_getter)
{
  auto mat = mfmat::dense_matrix<std::int64_t, 2, 2>(mfmat::identity_tag());
  std::int64_t cell = mat.get<0, 0>();
  BOOST_CHECK(cell == 1);
  cell = mat.get<1, 0>();
  BOOST_CHECK(cell == 0);
  cell = mat.get<0, 1>();
  BOOST_CHECK(cell == 0);
  cell = mat.get<1, 1>();
  BOOST_CHECK(cell == 1);
}


BOOST_AUTO_TEST_CASE(compile_time_scan_rows_getter)
{
  auto mat = mfmat::dense_matrix<std::int64_t, 2, 3>
    ({
      { 1, 2, 3 },
      { 4, 5, 6 }
     });
  BOOST_CHECK(mat.scan_r<0>() == 1);
  BOOST_CHECK(mat.scan_r<1>() == 2);
  BOOST_CHECK(mat.scan_r<2>() == 3);
  BOOST_CHECK(mat.scan_r<3>() == 4);
  BOOST_CHECK(mat.scan_r<4>() == 5);
  BOOST_CHECK(mat.scan_r<5>() == 6);
}


BOOST_AUTO_TEST_CASE(compile_time_scan_columns_getter)
{
  auto mat = mfmat::dense_matrix<std::int64_t, 2, 3>
    ({
      { 1, 2, 3 },
      { 4, 5, 6 }
     });
  BOOST_CHECK(mat.scan_c<0>() == 1);
  BOOST_CHECK(mat.scan_c<1>() == 4);
  BOOST_CHECK(mat.scan_c<2>() == 2);
  BOOST_CHECK(mat.scan_c<3>() == 5);
  BOOST_CHECK(mat.scan_c<4>() == 3);
  BOOST_CHECK(mat.scan_c<5>() == 6);
}


BOOST_AUTO_TEST_CASE(compile_time_scan_double_getter)
{
  auto mat = mfmat::dense_matrix<double, 2, 3>
    ({
      { 1.1, 2.2, 3.3 },
      { 4.4, 5.5, 6.6 }
     });
  BOOST_CHECK_CLOSE(mat.scan_r<0>(), 1.1, 0.0000001);
  BOOST_CHECK_CLOSE(mat.scan_r<1>(), 2.2, 0.0000001);
  BOOST_CHECK_CLOSE(mat.scan_r<2>(), 3.3, 0.0000001);
  BOOST_CHECK_CLOSE(mat.scan_r<3>(), 4.4, 0.0000001);
  BOOST_CHECK_CLOSE(mat.scan_r<4>(), 5.5, 0.0000001);
  BOOST_CHECK_CLOSE(mat.scan_r<5>(), 6.6, 0.0000001);
  BOOST_CHECK_CLOSE(mat.scan_c<0>(), 1.1, 0.0000001);
  BOOST_CHECK_CLOSE(mat.scan_c<1>(), 4.4, 0.0000001);
  BOOST_CHECK_CLOSE(mat.scan_c<2>(), 2.2, 0.0000001);
  BOOST_CHECK_CLOSE(mat.scan_c<3>(), 5.5, 0.0000001);
  BOOST_CHECK_CLOSE(mat.scan_c<4>(), 3.3, 0.0000001);
  BOOST_CHECK_CLOSE(mat.scan_c<5>(), 6.6, 0.0000001);
}

BOOST_AUTO_TEST_SUITE_END()
