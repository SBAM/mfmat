#include <sstream>

#include <mfmat/cl_kernels_store.hpp>

#define DECL_BINARY_RESOURCE(name)             \
  extern const char _binary_##name##_cl_start; \
  extern const char _binary_##name##_cl_end

DECL_BINARY_RESOURCE(prefix_float);
DECL_BINARY_RESOURCE(prefix_double);
DECL_BINARY_RESOURCE(matrix_scalar_add);
DECL_BINARY_RESOURCE(matrix_scalar_sub);
DECL_BINARY_RESOURCE(matrix_scalar_mul);
DECL_BINARY_RESOURCE(matrix_scalar_div);
DECL_BINARY_RESOURCE(matrix_add);
DECL_BINARY_RESOURCE(matrix_add_column);
DECL_BINARY_RESOURCE(matrix_add_row);
DECL_BINARY_RESOURCE(matrix_sub);
DECL_BINARY_RESOURCE(matrix_sub_column);
DECL_BINARY_RESOURCE(matrix_sub_row);
DECL_BINARY_RESOURCE(matrix_is_equal);
DECL_BINARY_RESOURCE(matrix_is_diagonal);
DECL_BINARY_RESOURCE(matrix_is_symmetric);
DECL_BINARY_RESOURCE(matrix_square_transpose);
DECL_BINARY_RESOURCE(matrix_orthonormalize);
DECL_BINARY_RESOURCE(matrix_transpose);
DECL_BINARY_RESOURCE(matrix_multiply);
DECL_BINARY_RESOURCE(matrix_mean);
DECL_BINARY_RESOURCE(matrix_mean_center);
DECL_BINARY_RESOURCE(matrix_stddev);
DECL_BINARY_RESOURCE(matrix_stddev_center);
DECL_BINARY_RESOURCE(matrix_self_multiply_regularized);
DECL_BINARY_RESOURCE(matrix_transpose_multiply);


#define STR_BINARY_RESOURCE(name)           \
  std::string(&_binary_##name##_cl_start,   \
              &_binary_##name##_cl_end -    \
              &_binary_##name##_cl_start)

#define INIT_MEMBER(name)                                   \
  name                                                      \
  {                                                         \
    .src_ = STR_BINARY_RESOURCE(name),                      \
    .f =                                                    \
    {                                                       \
      .prog_ = make_program(ctx, prefix_f_src_, name.src_), \
      .func_ = { name.f.prog_, #name }                      \
    },                                                      \
    .d =                                                    \
    {                                                       \
      .prog_ = make_program(ctx, prefix_d_src_, name.src_), \
      .func_ = { name.d.prog_, #name }                      \
    }                                                       \
  }


namespace mfmat
{

  cl_kernels_store::cl_kernels_store(context_opt ctx) :
    prefix_f_src_(STR_BINARY_RESOURCE(prefix_float)),
    prefix_d_src_(STR_BINARY_RESOURCE(prefix_double)),
    INIT_MEMBER(matrix_scalar_add),
    INIT_MEMBER(matrix_scalar_sub),
    INIT_MEMBER(matrix_scalar_mul),
    INIT_MEMBER(matrix_scalar_div),
    INIT_MEMBER(matrix_add),
    INIT_MEMBER(matrix_add_column),
    INIT_MEMBER(matrix_add_row),
    INIT_MEMBER(matrix_sub),
    INIT_MEMBER(matrix_sub_column),
    INIT_MEMBER(matrix_sub_row),
    INIT_MEMBER(matrix_is_equal),
    INIT_MEMBER(matrix_is_diagonal),
    INIT_MEMBER(matrix_is_symmetric),
    INIT_MEMBER(matrix_square_transpose),
    INIT_MEMBER(matrix_orthonormalize),
    INIT_MEMBER(matrix_transpose),
    INIT_MEMBER(matrix_multiply),
    INIT_MEMBER(matrix_mean),
    INIT_MEMBER(matrix_mean_center),
    INIT_MEMBER(matrix_stddev),
    INIT_MEMBER(matrix_stddev_center),
    INIT_MEMBER(matrix_self_multiply_regularized),
    INIT_MEMBER(matrix_transpose_multiply)
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
      result.build("-Werror");
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
