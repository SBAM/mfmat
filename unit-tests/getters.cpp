#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(getters_test_suite)

constexpr auto owr = mfmat::op_way::row;
constexpr auto owc = mfmat::op_way::col;

BOOST_AUTO_TEST_CASE(runtime_getter)
{
  auto mat = mfmat::ct_mat<std::int32_t, 5, 10>();
  for (std::size_t i = 0; i < mat.row_count; ++i)
    for (std::size_t j = 0; j < mat.col_count; ++j)
    {
      auto cell = mat[{i, j}];
      BOOST_CHECK(cell == 0);
    }
}


BOOST_AUTO_TEST_CASE(compile_time_getter)
{
  auto mat = mfmat::ct_mat<std::int64_t, 2, 2>(mfmat::identity_tag());
  std::int64_t res{};
  res = mat.get<0, 0>(); BOOST_CHECK(res == 1);
  res = mat.get<1, 0>(); BOOST_CHECK(res == 0);
  res = mat.get<0, 1>(); BOOST_CHECK(res == 0);
  res = mat.get<1, 1>(); BOOST_CHECK(res == 1);
}


BOOST_AUTO_TEST_CASE(compile_time_getter_with_way_specifier)
{
  auto mat = mfmat::ct_mat<std::int64_t, 2, 3>
    ({
      { 1, 2, 3 },
      { 4, 5, 6 }
     });
  std::int64_t res{};
  res = mat.get<owr, 0, 0>(); BOOST_CHECK(res == 1);
  res = mat.get<owr, 0, 1>(); BOOST_CHECK(res == 2);
  res = mat.get<owr, 0, 2>(); BOOST_CHECK(res == 3);
  res = mat.get<owr, 1, 0>(); BOOST_CHECK(res == 4);
  res = mat.get<owr, 1, 1>(); BOOST_CHECK(res == 5);
  res = mat.get<owr, 1, 2>(); BOOST_CHECK(res == 6);
  res = mat.get<owc, 0, 0>(); BOOST_CHECK(res == 1);
  res = mat.get<owc, 1, 0>(); BOOST_CHECK(res == 2);
  res = mat.get<owc, 2, 0>(); BOOST_CHECK(res == 3);
  res = mat.get<owc, 0, 1>(); BOOST_CHECK(res == 4);
  res = mat.get<owc, 1, 1>(); BOOST_CHECK(res == 5);
  res = mat.get<owc, 2, 1>(); BOOST_CHECK(res == 6);
}


BOOST_AUTO_TEST_CASE(compile_time_scan_rows_getter)
{
  auto mat = mfmat::ct_mat<std::int64_t, 2, 3>
    ({
      { 1, 2, 3 },
      { 4, 5, 6 }
     });
  std::int64_t res{};
  res = mat.scan<owr, 0>(); BOOST_CHECK(res == 1);
  res = mat.scan<owr, 1>(); BOOST_CHECK(res == 2);
  res = mat.scan<owr, 2>(); BOOST_CHECK(res == 3);
  res = mat.scan<owr, 3>(); BOOST_CHECK(res == 4);
  res = mat.scan<owr, 4>(); BOOST_CHECK(res == 5);
  res = mat.scan<owr, 5>(); BOOST_CHECK(res == 6);
}


BOOST_AUTO_TEST_CASE(compile_time_scan_columns_getter)
{
  auto mat = mfmat::ct_mat<std::int64_t, 2, 3>
    ({
      { 1, 2, 3 },
      { 4, 5, 6 }
     });
  std::int64_t res{};
  res = mat.scan<owc, 0>(); BOOST_CHECK(res == 1);
  res = mat.scan<owc, 1>(); BOOST_CHECK(res == 4);
  res = mat.scan<owc, 2>(); BOOST_CHECK(res == 2);
  res = mat.scan<owc, 3>(); BOOST_CHECK(res == 5);
  res = mat.scan<owc, 4>(); BOOST_CHECK(res == 3);
  res = mat.scan<owc, 5>(); BOOST_CHECK(res == 6);
}


BOOST_AUTO_TEST_CASE(compile_time_scan_double_getter)
{
  auto mat = mfmat::ct_mat<double, 2, 3>
    ({
      { 1.1, 2.2, 3.3 },
      { 4.4, 5.5, 6.6 }
     });
  double res{};
  res = mat.scan<owr, 0>(); BOOST_CHECK_CLOSE(res, 1.1, 0.0000001);
  res = mat.scan<owr, 1>(); BOOST_CHECK_CLOSE(res, 2.2, 0.0000001);
  res = mat.scan<owr, 2>(); BOOST_CHECK_CLOSE(res, 3.3, 0.0000001);
  res = mat.scan<owr, 3>(); BOOST_CHECK_CLOSE(res, 4.4, 0.0000001);
  res = mat.scan<owr, 4>(); BOOST_CHECK_CLOSE(res, 5.5, 0.0000001);
  res = mat.scan<owr, 5>(); BOOST_CHECK_CLOSE(res, 6.6, 0.0000001);
  res = mat.scan<owc, 0>(); BOOST_CHECK_CLOSE(res, 1.1, 0.0000001);
  res = mat.scan<owc, 1>(); BOOST_CHECK_CLOSE(res, 4.4, 0.0000001);
  res = mat.scan<owc, 2>(); BOOST_CHECK_CLOSE(res, 2.2, 0.0000001);
  res = mat.scan<owc, 3>(); BOOST_CHECK_CLOSE(res, 5.5, 0.0000001);
  res = mat.scan<owc, 4>(); BOOST_CHECK_CLOSE(res, 3.3, 0.0000001);
  res = mat.scan<owc, 5>(); BOOST_CHECK_CLOSE(res, 6.6, 0.0000001);
}

BOOST_AUTO_TEST_SUITE_END()
