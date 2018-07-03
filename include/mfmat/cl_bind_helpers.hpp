#ifndef MFMAT_CL_BIND_HELPERS_HPP_
# define MFMAT_CL_BIND_HELPERS_HPP_

# include "cl_kernels_store.hpp"

namespace mfmat
{

  /**
   * @return read/write CL's buffer with a vector bound to it
   * @tparam Ts vector's parameters
   * @param vec in/out vector
   */
  template <typename... Ts>
  inline cl::Buffer rw_bind(std::vector<Ts...>& vec);


  /**
   * @return read_only CL's buffer with a vector bound to it
   * @tparam Ts vector's parameters
   * @param vec RO in vector
   */
  template <typename... Ts>
  inline cl::Buffer ro_bind(const std::vector<Ts...>& vec);


  /**
   * @brief dispatches given arguments to specialized kernel
   * @tparam T specialized kernel's underlying type
   * @tparam K kernels store's kernel
   * @param ker_args kernel's arguments
   */
  template <typename T, typename K, typename... Args>
  inline void bind_ker(K& ker, Args&&... ker_args);


  /**
   * @brief Enqueues ouput read_buffer
   * @param clb CL's buffer
   * @param vec output storage
   */
  template <typename... Ts>
  inline void bind_res(const cl::Buffer& clb, std::vector<Ts...>& vec);

} // !namespace mfmat

# include "cl_bind_helpers.ipp"

#endif // !MFMAT_CL_BIND_HELPERS_HPP_
