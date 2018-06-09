#ifndef MFMAT_CL_DEFAULT_GPU_SETTER_HPP_
# define MFMAT_CL_DEFAULT_GPU_SETTER_HPP_

# define CL_HPP_MINIMUM_OPENCL_VERSION 120
# define CL_HPP_TARGET_OPENCL_VERSION 120
# define CL_HPP_CL_1_2_DEFAULT_BUILD
# define CL_HPP_ENABLE_EXCEPTIONS

# include <CL/cl2.hpp>

namespace mfmat
{

  /**
   * @brief Sets up default CL facilities to use first GPU found:
   *         - sets default platform
   *         - sets default device (first GPU found)
   *         - sets default context
   *         - sets default command queue
   */
  class cl_default_gpu_setter
  {
  public:
    /// @return static helper instance
    static cl_default_gpu_setter& instance()
    {
      static cl_default_gpu_setter local;
      return local;
    }

    /// @todo Provide getters to device descriptors

  private:
    /// @brief Sets up facilities, throws on failure
    cl_default_gpu_setter();
  };

} // !namespace mfmat

#endif // !MFMAT_CL_DEFAULT_GPU_SETTER_HPP_
