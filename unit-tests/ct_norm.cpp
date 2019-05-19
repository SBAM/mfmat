#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(norm_test_suite)

constexpr auto owr = mfmat::op_way::row;
constexpr auto owc = mfmat::op_way::col;

BOOST_AUTO_TEST_CASE(identity_1x1_norm)
{
  auto mat1 = mfmat::ct_mat<std::int32_t, 1, 1>(mfmat::identity_tag());
  auto norm1_1 = mat1.norm<owr, 0>();
  BOOST_CHECK_EQUAL(norm1_1, 1);
  auto norm1_2 = mat1.norm<owc, 0>();
  BOOST_CHECK_EQUAL(norm1_2, 1);
  auto mat2 = mfmat::ct_mat<double, 1, 1>({{2.0}});
  auto norm2_1 = mat2.norm<owr, 0>();
  BOOST_CHECK_CLOSE(norm2_1, 2.0, 0.001);
  auto norm2_2 = mat2.norm<owc, 0>();
  BOOST_CHECK_CLOSE(norm2_2, 2.0, 0.001);
}


BOOST_AUTO_TEST_CASE(norm_3x3)
{
  auto mat = mfmat::ct_mat<double, 3, 3>
    ({
      { 1.0, 2.0, 3.0 },
      { 2.0, 2.0, 4.0 },
      { 3.0, 4.0, 3.0 }
     });
  auto norm = mat.norm<owr, 0>();
  BOOST_CHECK_CLOSE(norm, std::sqrt(14.0), 0.001);
  norm = mat.norm<owr, 1>();
  BOOST_CHECK_CLOSE(norm, std::sqrt(24.0), 0.001);
  norm = mat.norm<owr, 2>();
  BOOST_CHECK_CLOSE(norm, std::sqrt(34.0), 0.001);
  norm = mat.norm<owc, 0>();
  BOOST_CHECK_CLOSE(norm, std::sqrt(14.0), 0.001);
  norm = mat.norm<owc, 1>();
  BOOST_CHECK_CLOSE(norm, std::sqrt(24.0), 0.001);
  norm = mat.norm<owc, 2>();
  BOOST_CHECK_CLOSE(norm, std::sqrt(34.0), 0.001);
}


BOOST_AUTO_TEST_CASE(norm_2x4)
{
  auto mat = mfmat::ct_mat<double, 2, 4>
    ({
      { 1.0, 2.0, 3.0, 4.0 },
      { 2.0, 3.0, 4.0, 5.0 }
     });
  auto norm = mat.norm<owr, 0>();
  BOOST_CHECK_CLOSE(norm, std::sqrt(30.0), 0.001);
  norm = mat.norm<owr, 1>();
  BOOST_CHECK_CLOSE(norm, std::sqrt(54.0), 0.001);
  norm = mat.norm<owc, 0>();
  BOOST_CHECK_CLOSE(norm, std::sqrt(5.0), 0.001);
  norm = mat.norm<owc, 1>();
  BOOST_CHECK_CLOSE(norm, std::sqrt(13.0), 0.001);
  norm = mat.norm<owc, 2>();
  BOOST_CHECK_CLOSE(norm, std::sqrt(25.0), 0.001);
  norm = mat.norm<owc, 3>();
  BOOST_CHECK_CLOSE(norm, std::sqrt(41.0), 0.001);
}

BOOST_AUTO_TEST_SUITE_END()
