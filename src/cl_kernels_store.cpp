#include <sstream>

#include <mfmat/cl_kernels_store.hpp>

#define DECL_BINARY_RESOURCE(name)             \
  extern const char _binary_##name##_cl_start; \
  extern const char _binary_##name##_cl_end

DECL_BINARY_RESOURCE(prefix_float);
DECL_BINARY_RESOURCE(prefix_double);
DECL_BINARY_RESOURCE(matrix_multiply);


#define INIT_BINARY_RESOURCE(name)        \
  name##_src_(&_binary_##name##_cl_start, \
              &_binary_##name##_cl_end -  \
              &_binary_##name##_cl_start)

#define INIT_DATA_TYPE_MEMBER(name, dt)                                    \
  name##_##dt##_prog_(make_program(ctx, prefix_##dt##_src_, name##_src_)), \
  name##_##dt(name##_##dt##_prog_, #name)

#define INIT_MEMBER(name)                                 \
  INIT_BINARY_RESOURCE(name),                             \
  INIT_DATA_TYPE_MEMBER(name, float),                     \
  INIT_DATA_TYPE_MEMBER(name, double)


namespace mfmat
{

  cl_kernels_store::cl_kernels_store(context_opt ctx) :
    INIT_BINARY_RESOURCE(prefix_float),
    INIT_BINARY_RESOURCE(prefix_double),
    INIT_MEMBER(matrix_multiply)
  {
  }


  cl::Program cl_kernels_store::make_program(context_opt ctx,
                                             const std::string& prefix,
                                             const std::string& src)
  {
    cl::Program result;
    if (ctx)
      result = cl::Program(*ctx, {prefix, src});
    else
      result = cl::Program({prefix, src});
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
        << prefix << src << std::endl
        << "------kernel_end-------" << std::endl
        << "err=[" << e.what() << " (" << e.err() << ")]" << std::endl
        << result.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl::Device::getDefault());
      throw std::runtime_error(err.str());
    }
    return result;
  }

} // !namespace mfmat
