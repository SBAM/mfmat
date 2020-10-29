#include <iomanip>
#include <iostream>
#include <sstream>

#include <mfmat/cl_default_gpu_setter.hpp>

namespace mfmat
{

  std::size_t cl_default_gpu_setter::get_max_work_group_size() const
  {
    return max_group_size_;
  }


  const cl_default_gpu_setter::ext_vec_t&
  cl_default_gpu_setter::get_extensions() const
  {
    return extensions_;
  }


  cl_default_gpu_setter::cl_default_gpu_setter()
  {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty())
      throw std::runtime_error("No CL platform available");
    for (const auto& curr_plat : platforms)
    {
      std::vector<cl::Device> devices;
      try
      {
        curr_plat.getDevices(CL_DEVICE_TYPE_GPU, &devices);
      }
      catch (const cl::Error&)
      {
        continue;
      }
      auto err_hdlr = [&](const std::string_view str_v, auto cb)
        {
          try
          {
            if (!cb())
            {
              std::ostringstream err;
              err << str_v;
              throw std::runtime_error(err.str());
            }
          }
          catch (const cl::Error& e)
          {
            std::ostringstream err;
            err
              << str_v
              << "[cl::Error code=" << e.err() << " msg=" << e.what() << ']';
            throw std::runtime_error(err.str());
          }
        };
      for (const auto& curr_dev : devices)
        if (curr_dev.getInfo<CL_DEVICE_AVAILABLE>())
        {
          cl::Platform default_platform;
          err_hdlr("Failed to set default CL platform",
                   [&]()
                   {
                     default_platform = cl::Platform::setDefault(curr_plat);
                     return default_platform == curr_plat;
                   });
          cl::Device default_device;
          err_hdlr("Failed to set default CL device",
                   [&]()
                   {
                     default_device = cl::Device::setDefault(curr_dev);
                     return default_device == curr_dev;
                   });
          cl::Context default_context;
          err_hdlr("Failed to set default CL context",
                   [&]()
                   {
                     auto tmp_context = cl::Context(default_device);
                     default_context = cl::Context::setDefault(tmp_context);
                     return default_context == tmp_context;
                   });
          cl::CommandQueue default_queue;
          err_hdlr("Failed to set default CL device command queue",
                   [&]()
                   {
                     auto tmp_queue = cl::CommandQueue
                       (default_context,
                        default_device,
                        CL_QUEUE_PROFILING_ENABLE);
                     default_queue = cl::CommandQueue::setDefault(tmp_queue);
                     return default_queue == tmp_queue;
                   });
          // retrieve hardware specific info for default GPU
          extract_device_info();
          return;
        }
    }
    throw std::runtime_error("No suitable CL environment was set");
  }


  void cl_default_gpu_setter::extract_device_info()
  {
    auto d = cl::Device::getDefault();
    vendor_ = d.getInfo<CL_DEVICE_VENDOR>();
    name_ = d.getInfo<CL_DEVICE_NAME>();
    driver_version_ = d.getInfo<CL_DRIVER_VERSION>();
    device_version_ = d.getInfo<CL_DEVICE_VERSION>();
    cl_version_ = d.getInfo<CL_DEVICE_OPENCL_C_VERSION>();
    compute_unit_ = d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    max_freq_ = d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
    global_mem_ = d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    global_mem_cache_ = d.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();
    local_mem_ = d.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    max_alloc_ = d.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    max_work_dim_ = d.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
    max_work_sizes_ = d.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    max_group_size_ = d.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    little_endian_ = d.getInfo<CL_DEVICE_ENDIAN_LITTLE>() == CL_TRUE;
    ts_res_ = d.getInfo<CL_DEVICE_PROFILING_TIMER_RESOLUTION>();
    image_support_ = d.getInfo<CL_DEVICE_IMAGE_SUPPORT>() == CL_TRUE;
    max_kernel_samplers_ = d.getInfo<CL_DEVICE_MAX_SAMPLERS>();
    // image1d_max_width_ = d.getInfo<CL_DEVICE_IMAGE_MAX_BUFFER_SIZE>();
    image2d_max_width_ = d.getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>();
    image2d_max_height_ = d.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>();
    image3d_max_width_ = d.getInfo<CL_DEVICE_IMAGE3D_MAX_WIDTH>();
    image3d_max_height_ = d.getInfo<CL_DEVICE_IMAGE3D_MAX_HEIGHT>();
    image3d_max_depth_ = d.getInfo<CL_DEVICE_IMAGE3D_MAX_DEPTH>();
    // split extensions
    std::istringstream ext_iss(d.getInfo<CL_DEVICE_EXTENSIONS>());
    using iss_it = std::istream_iterator<std::string>;
    extensions_ = ext_vec_t(iss_it(ext_iss), iss_it{});
    std::sort(extensions_.begin(), extensions_.end());
  }


  std::ostream& operator<<(std::ostream& os, const cl_default_gpu_setter& d)
  {
    constexpr auto kb = 1 << 10;
    constexpr auto mb = 1 << 20;
    os
      << "Default device:\n"
      << " vendor=" << d.vendor_ << '\n'
      << " name=" << d.name_ << '\n'
      << " driver_ver=" << d.driver_version_ << '\n'
      << " device_ver=" << d.device_version_ << '\n'
      << " cl_ver=" << d.cl_version_ << '\n'
      << " device_max_compute_units=" << d.compute_unit_ << '\n'
      << " device_max_clock_frequency=" << d.max_freq_ << "Mhz\n"
      << " device_global_memory=" << d.global_mem_ / mb << "MB\n"
      << " device_global_memory_cache=" << d.global_mem_cache_ / kb << "KB\n"
      << " device_local_memory=" << d.local_mem_ / kb << "KB\n"
      << " device_max_mem_alloc=" << d.max_alloc_ / mb << "MB\n"
      << " device_max_work_dim=" << d.max_work_dim_ << '\n'
      << " device_max_work_item_sizes=[ ";
    for (auto i : d.max_work_sizes_)
      os << i << ' ';
    os
      << std::boolalpha
      << "]\n"
      << " device_max_work_group_size=" << d.max_group_size_ << '\n'
      << " device_little_endian=" << d.little_endian_ << '\n'
      << " device_profiling_resolution=" << d.ts_res_ << "ns\n"
      << " device_image_support=" << d.image_support_ << '\n'
      << " device_max_kernel_samplers=" << d.max_kernel_samplers_ << '\n'
      // " device_image1D_max=[ " << d.image1d_max_width_ << " ]" << '\n'
      << " device_image2D_max=[ " << d.image2d_max_width_ << " x "
                                  << d.image2d_max_height_ << " ]\n"
      << " device_image3D_max=[ " << d.image3d_max_width_ << " x "
                                  << d.image3d_max_height_ << " x "
                                  << d.image3d_max_depth_ << " ]\n"
      << " extensions\n"
      << "  [\n";
    for (const auto& curr_ext : d.get_extensions())
      os << "   - " << curr_ext << '\n';
    os << "  ]";
    return os;
  }

} // !namespace mfmat
