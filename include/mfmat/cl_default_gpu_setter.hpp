#ifndef MFMAT_CL_DEFAULT_GPU_SETTER_HPP_
# define MFMAT_CL_DEFAULT_GPU_SETTER_HPP_

# define CL_TARGET_OPENCL_VERSION 200
# define CL_HPP_TARGET_OPENCL_VERSION 200
# define CL_HPP_MINIMUM_OPENCL_VERSION 200
# define CL_HPP_ENABLE_EXCEPTIONS

# include <CL/opencl.hpp>

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
    /// @typedef ext_vec_t Shorthand to extensions vector type
    using ext_vec_t = std::vector<std::string>;

  public:
    /// @return static helper instance
    static cl_default_gpu_setter& instance()
    {
      static cl_default_gpu_setter local;
      return local;
    }

    /// @return Max work group size
    std::size_t get_max_work_group_size() const;
    /// @return Extensions supported by device
    const ext_vec_t& get_extensions() const;

  private:
    /// @brief Sets up facilities, throws on failure
    cl_default_gpu_setter();
    /// @brief extracts and stores device description
    void extract_device_info();

  private:
    std::string vendor_; ///< device vendor
    std::string name_; ///< device name
    std::string driver_version_; ///< driver version
    std::string device_version_; ///< device version
    std::string cl_version_; ///< OpenCL version
    std::size_t compute_unit_; ///< device compute units
    std::size_t max_freq_; ///< device max clock frequency
    std::size_t global_mem_; ///< total global memory
    std::size_t global_mem_cache_; ///< global memory cache size
    std::size_t local_mem_; ///< total local memory
    std::size_t max_alloc_; ///< maximum memory allocation size
    std::size_t max_work_dim_; ///< max work items dimension
    std::vector<std::size_t> max_work_sizes_; ///< max work items per dimension
    std::size_t max_group_size_; ///< max work group size
    bool little_endian_; ///< true if device has little endian byte-order
    std::size_t ts_res_; ///< profiling event timestamp's resolution, in ns
    bool image_support_; ///< true if device has image support
    std::size_t max_kernel_samplers_; ///< max samplers per kernel
    // std::size_t image1d_max_width_; ///< image1D max width
    std::size_t image2d_max_width_; ///< image2D max width
    std::size_t image2d_max_height_; ///< image2D max height
    std::size_t image3d_max_width_; ///< image3D max width
    std::size_t image3d_max_height_; ///< image3D max height
    std::size_t image3d_max_depth_; ///< image3D max depth
    ext_vec_t extensions_; ///< supported extensions

    friend std::ostream&
    operator<<(std::ostream&, const cl_default_gpu_setter&);
  };

  /// @brief Dumps default device description
  std::ostream& operator<<(std::ostream&, const cl_default_gpu_setter&);

} // !namespace mfmat

#endif // !MFMAT_CL_DEFAULT_GPU_SETTER_HPP_
