#define BOOST_TEST_MODULE mfmat

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/tools/output_test_stream.hpp>

#include <mfmat/fwd.hpp>

BOOST_AUTO_TEST_SUITE(kernels_store_initialization_test_suite)

BOOST_AUTO_TEST_CASE(initialization)
{
  try
  {
    mfmat::cl_default_gpu_setter::instance();
  }
  catch (const std::exception& e)
  {
    std::cout << e.what() << std::endl;
  }
  BOOST_CHECK_NO_THROW(mfmat::cl_default_gpu_setter::instance());
  std::cout << mfmat::cl_default_gpu_setter::instance() << std::endl;
  try
  {
    mfmat::cl_kernels_store::instance();
  }
  catch (const std::exception& e)
  {
    std::cout << e.what() << std::endl;
  }
  BOOST_CHECK_NO_THROW(mfmat::cl_kernels_store::instance());
}

BOOST_AUTO_TEST_SUITE_END()
