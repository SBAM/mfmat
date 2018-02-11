#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(matrices_ops_test_suite)

BOOST_AUTO_TEST_CASE(add_and_store)
{
  auto mat = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { 1, 1, 1 },
      { 1, 1, 1 }
     });
  auto mat2 = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { 2, 2, 2 },
      { 2, 2, 2 }
     });
  mat += mat2;
  auto mat3 = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { 3, 3, 3 },
      { 3, 3, 3 }
     });
  BOOST_CHECK(mat == mat3);
}


BOOST_AUTO_TEST_CASE(substract_and_store)
{
  auto mat = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { 1, 1, 1 },
      { 1, 1, 1 }
     });
  auto mat2 = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { 2, 2, 2 },
      { 2, 2, 2 }
     });
  mat -= mat2;
  auto mat3 = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { -1, -1, -1 },
      { -1, -1, -1 }
     });
  BOOST_CHECK(mat == mat3);
}


BOOST_AUTO_TEST_CASE(add)
{
  auto mat = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { 1, 1, 1 },
      { 1, 1, 1 }
     });
  auto mat2 = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { 2, 2, 2 },
      { 2, 2, 2 }
     });
  auto res = mat + mat2;
  auto mat3 = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { 3, 3, 3 },
      { 3, 3, 3 }
     });
  BOOST_CHECK(res == mat3);
}


BOOST_AUTO_TEST_CASE(substract)
{
  auto mat = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { 1, 1, 1 },
      { 1, 1, 1 }
     });
  auto mat2 = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { 2, 2, 2 },
      { 2, 2, 2 }
     });
  auto res = mat - mat2;
  auto mat3 = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { -1, -1, -1 },
      { -1, -1, -1 }
     });
  BOOST_CHECK(res == mat3);
}


BOOST_AUTO_TEST_CASE(multiply_simple)
{
  auto mat = mfmat::dense_matrix<std::int32_t, 2, 3>
    ({
      { 1, 2, 3 },
      { 4, 5, 6 }
     });
  auto mat2 = mfmat::dense_matrix<std::int32_t, 3, 2>
    ({
      { 7, 8 },
      { 9, 10 },
      { 11, 12 }
     });
  auto mat3 = mfmat::dense_matrix<std::int32_t, 2, 2>
    ({
      { 58, 64 },
      { 139, 154 }
     });
  auto res = mat * mat2;
  BOOST_CHECK(res == mat3);
}


BOOST_AUTO_TEST_CASE(multiply_type_promotion)
{
  auto mat = mfmat::dense_matrix<std::int8_t, 1, 2>
    ({
      { 100, 100 }
     });
  auto mat2 = mfmat::dense_matrix<std::int32_t, 2, 1>
    ({
      { 100 },
      { 100 }
     });
  auto mat3 = mfmat::dense_matrix<std::int32_t, 1, 1>({ { 20000 } });
  auto res = mat * mat2;
  BOOST_CHECK(res == mat3);
}

BOOST_AUTO_TEST_SUITE_END()
