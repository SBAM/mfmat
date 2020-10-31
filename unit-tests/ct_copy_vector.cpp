#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/tools/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(copy_vector_test_suite)

constexpr auto owr = mfmat::op_way::row;
constexpr auto owc = mfmat::op_way::col;

BOOST_AUTO_TEST_CASE(all_rows_copy)
{
  auto src_mat_i = mfmat::ct_mat<std::int32_t, 3, 3>
    ({
      { 1, 2, 3},
      { 4, 5, 6},
      { 7, 8, 9}
     });
  auto dst_mat_i = mfmat::ct_mat<std::int32_t, 3, 3>();
  mfmat::copy_vector<owr, 0, owr, 0>(src_mat_i, dst_mat_i);
  mfmat::copy_vector<owr, 1, owr, 1>(src_mat_i, dst_mat_i);
  mfmat::copy_vector<owr, 2, owr, 2>(src_mat_i, dst_mat_i);
  BOOST_CHECK(src_mat_i == dst_mat_i);
  auto src_mat_d = mfmat::ct_mat<double, 2, 4>
    ({
      { 1.1, 2.2, 3.3, 4.4 },
      { 5.5, 6.6, 7.7, 8.8 }
     });
  auto dst_mat_d = mfmat::ct_mat<double, 2, 4>();
  mfmat::copy_vector<owr, 0, owr, 0>(src_mat_d, dst_mat_d);
  mfmat::copy_vector<owr, 1, owr, 1>(src_mat_d, dst_mat_d);
  BOOST_CHECK(src_mat_d == dst_mat_d);
}


BOOST_AUTO_TEST_CASE(all_columns_copy)
{
  auto src_mat_i = mfmat::ct_mat<std::int32_t, 3, 3>
    ({
      { 1, 2, 3},
      { 4, 5, 6},
      { 7, 8, 9}
     });
  auto dst_mat_i = mfmat::ct_mat<std::int32_t, 3, 3>();
  mfmat::copy_vector<owc, 0, owc, 0>(src_mat_i, dst_mat_i);
  mfmat::copy_vector<owc, 1, owc, 1>(src_mat_i, dst_mat_i);
  mfmat::copy_vector<owc, 2, owc, 2>(src_mat_i, dst_mat_i);
  BOOST_CHECK(src_mat_i == dst_mat_i);
  auto src_mat_d = mfmat::ct_mat<double, 2, 4>
    ({
      { 1.1, 2.2, 3.3, 4.4 },
      { 5.5, 6.6, 7.7, 8.8 }
     });
  auto dst_mat_d = mfmat::ct_mat<double, 2, 4>();
  mfmat::copy_vector<owc, 0, owc, 0>(src_mat_d, dst_mat_d);
  mfmat::copy_vector<owc, 1, owc, 1>(src_mat_d, dst_mat_d);
  mfmat::copy_vector<owc, 2, owc, 2>(src_mat_d, dst_mat_d);
  mfmat::copy_vector<owc, 3, owc, 3>(src_mat_d, dst_mat_d);
  BOOST_CHECK(src_mat_d == dst_mat_d);
}


BOOST_AUTO_TEST_CASE(copy_rows_to_columns)
{
  auto src_mat_i = mfmat::ct_mat<std::int32_t, 2, 4>
    ({
      { 1, 2, 3, 4 },
      { 5, 6, 7, 8 }
     });
  auto dst_mat_i = mfmat::ct_mat<std::int32_t, 4, 2>();
  mfmat::copy_vector<owr, 0, owc, 0>(src_mat_i, dst_mat_i);
  mfmat::copy_vector<owr, 1, owc, 1>(src_mat_i, dst_mat_i);
  BOOST_CHECK(transpose(src_mat_i) == dst_mat_i);
}


BOOST_AUTO_TEST_CASE(copy_columns_to_rows)
{
  auto src_mat_d = mfmat::ct_mat<double, 4, 2>
    ({
      { 1, 2 },
      { 3, 4 },
      { 5, 6 },
      { 7, 8 }
     });
  auto dst_mat_d = mfmat::ct_mat<double, 2, 4>();
  mfmat::copy_vector<owc, 0, owr, 0>(src_mat_d, dst_mat_d);
  mfmat::copy_vector<owc, 1, owr, 1>(src_mat_d, dst_mat_d);
  BOOST_CHECK(transpose(src_mat_d) == dst_mat_d);
}

BOOST_AUTO_TEST_SUITE_END()
