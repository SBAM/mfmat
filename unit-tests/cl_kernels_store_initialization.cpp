#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(kernels_store_initialization_test_suite)

BOOST_AUTO_TEST_CASE(initialization)
{
  BOOST_CHECK_NO_THROW(mfmat::cl_default_gpu_setter::instance());
  BOOST_CHECK_NO_THROW(mfmat::cl_kernels_store::instance());
}

BOOST_AUTO_TEST_SUITE_END()
