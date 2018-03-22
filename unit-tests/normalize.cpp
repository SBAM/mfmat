#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(normalize_test_suite)

constexpr auto owr = mfmat::op_way::row;
constexpr auto owc = mfmat::op_way::col;

BOOST_AUTO_TEST_CASE(normalize_identity)
{
  auto orig_i = mfmat::ct_mat<std::int32_t, 3, 3>(mfmat::identity_tag());
  auto mat_i = orig_i;
  auto success_i = true;
  success_i &= mat_i.normalize<owr, 0>();
  success_i &= mat_i.normalize<owr, 1>();
  success_i &= mat_i.normalize<owr, 2>();
  success_i &= mat_i.normalize<owc, 0>();
  success_i &= mat_i.normalize<owc, 1>();
  success_i &= mat_i.normalize<owc, 2>();
  BOOST_CHECK(orig_i == mat_i);
  BOOST_CHECK(success_i);
  auto orig_d = mfmat::ct_mat<double, 3, 3>(mfmat::identity_tag());
  auto mat_d = orig_d;
  auto success_d = true;
  success_d &= mat_d.normalize<owr, 0>();
  success_d &= mat_d.normalize<owr, 1>();
  success_d &= mat_d.normalize<owr, 2>();
  success_d &= mat_d.normalize<owc, 0>();
  success_d &= mat_d.normalize<owc, 1>();
  success_d &= mat_d.normalize<owc, 2>();
  BOOST_CHECK(orig_d == mat_d);
  BOOST_CHECK(success_d);
}


BOOST_AUTO_TEST_CASE(normalize_degenerate)
{
  auto orig_i = mfmat::ct_mat<std::int32_t, 2, 2>();
  auto mat_i = orig_i;
  auto success_i = false;
  success_i |= mat_i.normalize<owr, 0>();
  success_i |= mat_i.normalize<owr, 1>();
  success_i |= mat_i.normalize<owc, 0>();
  success_i |= mat_i.normalize<owc, 1>();
  BOOST_CHECK(orig_i == mat_i);
  BOOST_CHECK(!success_i);
  auto orig_d = mfmat::ct_mat<double, 2, 2>();
  auto mat_d = orig_d;
  auto success_d = false;
  success_d |= mat_d.normalize<owr, 0>();
  success_d |= mat_d.normalize<owr, 1>();
  success_d |= mat_d.normalize<owc, 0>();
  success_d |= mat_d.normalize<owc, 1>();
  BOOST_CHECK(orig_d == mat_d);
  BOOST_CHECK(!success_d);
}


BOOST_AUTO_TEST_CASE(normalize_2x4_rows)
{
  auto mat = mfmat::ct_mat<double, 2, 4>
    ({
      { 1.0, 2.0, 3.0, 4.0 },
      { 2.0, 3.0, 4.0, 5.0 }
     });
  auto success = true;
  success &= mat.normalize<owr, 0>();
  success &= mat.normalize<owr, 1>();
  double cell {};
  auto norm1 = std::sqrt(30.0);
  cell = mat.get<0, 0>(); BOOST_CHECK_CLOSE(cell, 1.0 / norm1, 0.0000001);
  cell = mat.get<0, 1>(); BOOST_CHECK_CLOSE(cell, 2.0 / norm1, 0.0000001);
  cell = mat.get<0, 2>(); BOOST_CHECK_CLOSE(cell, 3.0 / norm1, 0.0000001);
  cell = mat.get<0, 3>(); BOOST_CHECK_CLOSE(cell, 4.0 / norm1, 0.0000001);
  auto norm2 = std::sqrt(54.0);
  cell = mat.get<1, 0>(); BOOST_CHECK_CLOSE(cell, 2.0 / norm2, 0.0000001);
  cell = mat.get<1, 1>(); BOOST_CHECK_CLOSE(cell, 3.0 / norm2, 0.0000001);
  cell = mat.get<1, 2>(); BOOST_CHECK_CLOSE(cell, 4.0 / norm2, 0.0000001);
  cell = mat.get<1, 3>(); BOOST_CHECK_CLOSE(cell, 5.0 / norm2, 0.0000001);
  BOOST_CHECK(success);
}


BOOST_AUTO_TEST_CASE(norm_2x4_colums)
{
  auto mat = mfmat::ct_mat<double, 2, 4>
    ({
      { 1.0, 2.0, 3.0, 4.0 },
      { 2.0, 3.0, 4.0, 5.0 }
     });
  auto success = true;
  success &= mat.normalize<owc, 0>();
  success &= mat.normalize<owc, 1>();
  success &= mat.normalize<owc, 2>();
  success &= mat.normalize<owc, 3>();
  double cell {};
  auto norm1 = std::sqrt(5.0);
  cell = mat.get<0, 0>(); BOOST_CHECK_CLOSE(cell, 1.0 / norm1, 0.0000001);
  cell = mat.get<1, 0>(); BOOST_CHECK_CLOSE(cell, 2.0 / norm1, 0.0000001);
  auto norm2 = std::sqrt(13.0);
  cell = mat.get<0, 1>(); BOOST_CHECK_CLOSE(cell, 2.0 / norm2, 0.0000001);
  cell = mat.get<1, 1>(); BOOST_CHECK_CLOSE(cell, 3.0 / norm2, 0.0000001);
  auto norm3 = std::sqrt(25.0);
  cell = mat.get<0, 2>(); BOOST_CHECK_CLOSE(cell, 3.0 / norm3, 0.0000001);
  cell = mat.get<1, 2>(); BOOST_CHECK_CLOSE(cell, 4.0 / norm3, 0.0000001);
  auto norm4 = std::sqrt(41.0);
  cell = mat.get<0, 3>(); BOOST_CHECK_CLOSE(cell, 4.0 / norm4, 0.0000001);
  cell = mat.get<1, 3>(); BOOST_CHECK_CLOSE(cell, 5.0 / norm4, 0.0000001);
}

BOOST_AUTO_TEST_SUITE_END()
