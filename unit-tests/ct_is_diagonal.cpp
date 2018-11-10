#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(ct_is_diagonal_test_suite)

BOOST_AUTO_TEST_CASE(is_diagonal_1)
{
  auto mat1 = mfmat::ct_mat<std::int32_t, 1, 1>();
  BOOST_CHECK(mat1.is_diagonal());
  auto mat2 = mfmat::ct_mat<std::int32_t, 1, 1>(mfmat::identity_tag{});
  BOOST_CHECK(mat2.is_diagonal());
  auto mat3 = mfmat::ct_mat<std::int32_t, 2, 2>();
  BOOST_CHECK(mat3.is_diagonal());
  auto mat4 = mfmat::ct_mat<std::int32_t, 2, 2>(mfmat::identity_tag{});
  BOOST_CHECK(mat4.is_diagonal());
  mat4 *= 2;
  BOOST_CHECK(mat4.is_diagonal());
  auto mat5 = mfmat::ct_mat<std::int32_t, 3, 3>
    ({
      { 1, 0, 0 },
      { 0, 2, 0 },
      { 0, 0, 3 }
     });
  BOOST_CHECK(mat5.is_diagonal());
  auto mat6 = mfmat::ct_mat<std::int32_t, 3, 4>
    ({
      { 1, 0, 0, 0 },
      { 0, 2, 0, 0 },
      { 0, 0, 3, 0 }
     });
  BOOST_CHECK(mat6.is_diagonal());
  auto mat7 = mfmat::ct_mat<std::int32_t, 4, 3>
    ({
      { 1, 0, 0 },
      { 0, 2, 0 },
      { 0, 0, 3 },
      { 0, 0, 0 }
     });
  BOOST_CHECK(mat7.is_diagonal());
}


BOOST_AUTO_TEST_CASE(is_diagonal_2)
{
  auto mat1 = mfmat::ct_mat<std::int32_t, 2, 2>
    ({
      { 1, 1 },
      { 0, 1 }
     });
  BOOST_CHECK(!mat1.is_diagonal());
  auto mat2 = mfmat::ct_mat<std::int32_t, 3, 3>
    ({
      { 1, 0, 0 },
      { 0, 2, 0 },
      { 1, 0, 3 }
     });
  BOOST_CHECK(!mat2.is_diagonal());
  auto mat3 = mfmat::ct_mat<std::int32_t, 3, 4>
    ({
      { 1, 0, 0, 1 },
      { 0, 2, 0, 0 },
      { 0, 0, 3, 0 }
     });
  BOOST_CHECK(!mat3.is_diagonal());
  auto mat4 = mfmat::ct_mat<std::int32_t, 4, 3>
    ({
      { 1, 0, 0 },
      { 0, 2, 0 },
      { 0, 0, 3 },
      { 1, 0, 0 }
     });
  BOOST_CHECK(!mat4.is_diagonal());
}


BOOST_AUTO_TEST_CASE(is_diagonal_3)
{
  auto mat1 = mfmat::ct_mat<double, 1, 1>();
  BOOST_CHECK(mat1.is_diagonal());
  BOOST_CHECK(mat1.is_diagonal(2.0));
  auto mat2 = mfmat::ct_mat<double, 1, 1>(mfmat::identity_tag{});
  BOOST_CHECK(mat2.is_diagonal());
  BOOST_CHECK(mat2.is_diagonal(2.0));
  auto mat3 = mfmat::ct_mat<double, 2, 2>();
  BOOST_CHECK(mat3.is_diagonal());
  BOOST_CHECK(mat3.is_diagonal(2.0));
  auto mat4 = mfmat::ct_mat<double, 2, 2>(mfmat::identity_tag{});
  BOOST_CHECK(mat4.is_diagonal());
  BOOST_CHECK(mat4.is_diagonal(2.0));
  mat4 *= 2.0;
  BOOST_CHECK(mat4.is_diagonal());
  BOOST_CHECK(mat4.is_diagonal(2.0));
  auto mat5 = mfmat::ct_mat<double, 3, 3>
    ({
      { 1.0, 0.0, 0.0 },
      { 0.0, 2.0, 0.0 },
      { 0.0, 0.0, 3.0 }
     });
  BOOST_CHECK(mat5.is_diagonal());
  BOOST_CHECK(mat5.is_diagonal(2.0));
  auto mat6 = mfmat::ct_mat<double, 3, 4>
    ({
      { 1.0, 0.0, 0.0, 0.0 },
      { 0.0, 2.0, 0.0, 0.0 },
      { 0.0, 0.0, 3.0, 0.0 }
     });
  BOOST_CHECK(mat6.is_diagonal());
  BOOST_CHECK(mat6.is_diagonal(2.0));
  auto mat7 = mfmat::ct_mat<double, 4, 3>
    ({
      { 1.0, 0.0, 0.0 },
      { 0.0, 2.0, 0.0 },
      { 0.0, 0.0, 3.0 },
      { 0.0, 0.0, 0.0 }
     });
  BOOST_CHECK(mat7.is_diagonal());
  BOOST_CHECK(mat7.is_diagonal(2.0));
}


BOOST_AUTO_TEST_CASE(is_diagonal_4)
{
  auto mat1 = mfmat::ct_mat<double, 2, 2>
    ({
      { 1.0, 1.0 },
      { 0.0, 1.0 }
     });
  BOOST_CHECK(!mat1.is_diagonal());
  BOOST_CHECK(!mat1.is_diagonal(2.0));
  auto mat2 = mfmat::ct_mat<double, 3, 3>
    ({
      { 1.0, 0.0, 0.0 },
      { 0.0, 2.0, 0.0 },
      { 1.0, 0.0, 3.0 }
     });
  BOOST_CHECK(!mat2.is_diagonal());
  BOOST_CHECK(!mat2.is_diagonal(2.0));
  auto mat3 = mfmat::ct_mat<double, 3, 4>
    ({
      { 1.0, 0.0, 0.0, 1.0 },
      { 0.0, 2.0, 0.0, 0.0 },
      { 0.0, 0.0, 3.0, 0.0 }
     });
  BOOST_CHECK(!mat3.is_diagonal());
  BOOST_CHECK(!mat3.is_diagonal(2.0));
  auto mat4 = mfmat::ct_mat<double, 4, 3>
    ({
      { 1.0, 0.0, 0.0 },
      { 0.0, 2.0, 0.0 },
      { 0.0, 0.0, 3.0 },
      { 1.0, 0.0, 0.0 }
     });
  BOOST_CHECK(!mat4.is_diagonal());
  BOOST_CHECK(!mat4.is_diagonal(2.0));
}


BOOST_AUTO_TEST_CASE(is_diagonal_5)
{
  static constexpr auto eps = std::numeric_limits<double>::epsilon();
  auto mat1 = mfmat::ct_mat<double, 3, 3>
    ({
      { 1.0, eps, eps },
      { eps, 2.0, eps },
      { eps, eps, 3.0 }
     });
  BOOST_CHECK(mat1.is_diagonal());
  BOOST_CHECK(mat1.is_diagonal(2.0));
  auto mat2 = mat1 * 2.0;
  BOOST_CHECK(!mat2.is_diagonal());
  BOOST_CHECK(mat2.is_diagonal(2.0));
  auto mat3 = mat1 * 2.5;
  BOOST_CHECK(!mat3.is_diagonal());
  BOOST_CHECK(!mat3.is_diagonal(2.0));
  BOOST_CHECK(mat3.is_diagonal(3.0));
}

BOOST_AUTO_TEST_SUITE_END()
