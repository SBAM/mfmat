#ifndef MFMAT_CL_KERNELS_STORE_HPP_
# define MFMAT_CL_KERNELS_STORE_HPP_

# include <optional>

# include "cl_default_gpu_setter.hpp"

# define DECL_MEMBER(name, ...)                           \
  const std::string name##_src_;                          \
  const cl::Program name##_prog_;                         \
  using name##_func_t = cl::KernelFunctor<__VA_ARGS__>;   \
  name##_func_t name

namespace mfmat
{

  /**
   * @brief Stores for each CL kernel:
   *         - its source
   *         - its compiled program
   *         - its invokable helper functor
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
      /// @todo std::call_once cl_default_gpu_setter if ctx = nullopt
      static cl_kernels_store local(ctx);
      return local;
    }

    /**
     * @brief Wraps Program construction and reformats thrown exception with
     *        additional details upon compilation failure.
     * @param ctx Context to use
     * @param src Program source
     * @return compiled Program
     */
    static cl::Program make_program(context_opt ctx, const std::string& src);


    DECL_MEMBER(matrix_multiply,
                std::size_t, std::size_t, std::size_t,
                cl::Buffer, cl::Buffer, cl::Buffer);
  };

} // !namespace mfmat

#endif // !MFMAT_CL_KERNELS_STORE_HPP_
