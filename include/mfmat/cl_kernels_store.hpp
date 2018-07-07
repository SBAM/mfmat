#ifndef MFMAT_CL_KERNELS_STORE_HPP_
# define MFMAT_CL_KERNELS_STORE_HPP_

# include <optional>

# include "cl_default_gpu_setter.hpp"

# define DECL_KERNEL(name, ...)                      \
  struct name##_t                                    \
  {                                                  \
    const std::string src_;                          \
    struct f_t                                       \
    {                                                \
      using data_t = float;                          \
      using func_t = cl::KernelFunctor<__VA_ARGS__>; \
      const cl::Program prog_;                       \
      func_t func_;                                  \
    };                                               \
    f_t f;                                           \
    struct d_t                                       \
    {                                                \
      using data_t = double;                         \
      using func_t = cl::KernelFunctor<__VA_ARGS__>; \
      const cl::Program prog_;                       \
      func_t func_;                                  \
    };                                               \
    d_t d;                                           \
  };                                                 \
  name##_t name


namespace mfmat
{

  /**
   * @brief Stores for each CL kernel:
   *         - its source
   *         - its compiled program for both float and double data types
   *         - its invokable helper functors for each data types
   */
  struct cl_kernels_store
  {
    /// @typedef context_opt shorthand to optional CL Context
    using context_opt = std::optional<cl::Context>;

    /**
     * @brief Sets up internal facilities and compiles programs using given
     *        context.
     * @param ctx optional context that overrides default context
     */
    cl_kernels_store(context_opt ctx = std::nullopt);


    /// @return static store instance
    static cl_kernels_store& instance(context_opt ctx = std::nullopt)
    {
      static cl_kernels_store local(ctx);
      return local;
    }

    /**
     * @brief Wraps Program construction and reformats thrown exception with
     *        additional details upon compilation failure.
     * @param ctx Context to use
     * @param prefix Specifies kernel defines
     * @param src Program source
     * @return compiled Program
     */
    static cl::Program make_program(context_opt ctx,
                                    const std::string& prefix,
                                    const std::string& src);

    const std::string prefix_f_src_; ///< prefixes float kernels
    const std::string prefix_d_src_; ///< prefixes double kernels
    DECL_KERNEL(matrix_scalar_add, cl::Buffer, data_t);
    DECL_KERNEL(matrix_scalar_sub, cl::Buffer, data_t);
    DECL_KERNEL(matrix_scalar_mul, cl::Buffer, data_t);
    DECL_KERNEL(matrix_scalar_div, cl::Buffer, data_t);
    DECL_KERNEL(matrix_add, cl::Buffer, cl::Buffer);
    DECL_KERNEL(matrix_add_column, cl::Buffer, std::size_t, cl::Buffer);
    DECL_KERNEL(matrix_add_row, cl::Buffer, std::size_t, cl::Buffer);
    DECL_KERNEL(matrix_sub, cl::Buffer, cl::Buffer);
    DECL_KERNEL(matrix_sub_column, cl::Buffer, std::size_t, cl::Buffer);
    DECL_KERNEL(matrix_sub_row, cl::Buffer, std::size_t, cl::Buffer);
    DECL_KERNEL(matrix_square_transpose, cl::Buffer, std::size_t, std::size_t);
    DECL_KERNEL(matrix_transpose, cl::Buffer, std::size_t, std::size_t,
                cl::Buffer);
    DECL_KERNEL(matrix_multiply, std::size_t, std::size_t, std::size_t,
                cl::Buffer, cl::Buffer, cl::Buffer);
  };

} // !namespace mfmat

# undef DECL_KERNEL

#endif // !MFMAT_CL_KERNELS_STORE_HPP_
