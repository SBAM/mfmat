#ifndef MFMAT_CL_KERNELS_STORE_HPP_
# define MFMAT_CL_KERNELS_STORE_HPP_

# include <optional>

# include "cl_default_gpu_setter.hpp"

# define DECL_MEMBER(name, ...)                           \
  const std::string name##_src_;                          \
  using name##_func_t = cl::KernelFunctor<__VA_ARGS__>;   \
  const cl::Program name##_float_prog_;                   \
  name##_func_t name##_float;                             \
  const cl::Program name##_double_prog_;                  \
  name##_func_t name##_double

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

    const std::string prefix_float_src_; ///< prefixes float kernels
    const std::string prefix_double_src_; ///< prefixes double kernels
    DECL_MEMBER(matrix_multiply,
                std::size_t, std::size_t, std::size_t,
                cl::Buffer, cl::Buffer, cl::Buffer);
  };

} // !namespace mfmat

#endif // !MFMAT_CL_KERNELS_STORE_HPP_
