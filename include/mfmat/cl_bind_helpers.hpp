#ifndef MFMAT_CL_BIND_HELPERS_HPP_
# define MFMAT_CL_BIND_HELPERS_HPP_

# include "cl_kernels_store.hpp"

namespace mfmat
{

  /**
   * @return GPU can read and write to CL's buffer
   * @tparam Ts vector's parameters
   * @param vec in/out vector
   */
  template <typename... Ts>
  inline cl::Buffer rw_bind(std::vector<Ts...>& vec);


  /**
   * @return GPU can read_only CL's buffer
   * @tparam Ts vector's parameters
   * @param vec RO in vector
   */
  template <typename... Ts>
  inline cl::Buffer ro_bind(const std::vector<Ts...>& vec);


  /**
   * @return GPU can write_only to CL's buffer, only size to allocate on device
   *         needs to be specified
   * @param len bytes to allocate on device
   */
  inline cl::Buffer wo_bind(std::size_t len);


  /**
   * @return GPU can write_only to CL's buffer defined by input vector length
   * @tparam Ts vector's parameters
   * @param vec WO in vector
   */
  template <typename... Ts>
  inline cl::Buffer wo_bind(const std::vector<Ts...>& vec);


  /**
   * @return Only GPU can read and write to CL's buffer, only size to allocate
   *         on device needs to be specified
   * @param len bytes to allocate on device
   */
  inline cl::Buffer no_host_bind(std::size_t len);


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
   * @return cl::Event filled following read buffer, forwarded from
   *         cl::KernelFunctor's call
   */
  template <typename... Ts>
  inline cl::Event bind_res(const cl::Buffer& clb, std::vector<Ts...>& vec);


  /**
   * @brief Enqueues ouput read_buffer
   * @param clb CL's buffer
   * @param ptr output region
   * @param elt_count element count
   * @return cl::Event filled following read buffer, forwarded from
   *         cl::KernelFunctor's call
   */
  template <typename T>
  inline cl::Event bind_res(const cl::Buffer& clb, T* ptr, std::size_t elt_count);

} // !namespace mfmat

# include "cl_bind_helpers.ipp"

#endif // !MFMAT_CL_BIND_HELPERS_HPP_
