#include <sstream>

#include <mfmat/cl_kernels_store.hpp>

#define DECL_BINARY_RESOURCE(name)             \
  extern const char _binary_##name##_cl_start; \
  extern const char _binary_##name##_cl_end

DECL_BINARY_RESOURCE(matrix_multiply);


#define INIT_MEMBER(name)                       \
  name##_src_(&_binary_##name##_cl_start,       \
              &_binary_##name##_cl_end -        \
              &_binary_##name##_cl_start),      \
  name##_prog_(make_program(ctx, name##_src_)), \
  name(name##_prog_, #name)


namespace mfmat
{

  cl_kernels_store::cl_kernels_store(context_opt ctx) :
    INIT_MEMBER(matrix_multiply)
  {
  }


  cl::Program
  cl_kernels_store::make_program(context_opt ctx, const std::string& src)
  {
    cl::Program result;
    if (ctx)
      result = cl::Program(*ctx, src);
    else
      result = cl::Program(src);
    try
    {
      result.build();
    }
    catch (const cl::Error& e)
    {
      std::ostringstream err;
      err
        << "OpenCL compilation error for kernel=" << std::endl
        << "-----kernel_start------" << std::endl
        << src << std::endl
        << "------kernel_end-------" << std::endl
        << "err=[" << e.what() << " (" << e.err() << ")]" << std::endl
        << result.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl::Device::getDefault());
      throw std::runtime_error(err.str());
    }
    return result;
  }

} // !namespace mfmat
