#include <iomanip>
#include <sstream>

#include <mfmat/cl_default_gpu_setter.hpp>

namespace mfmat
{

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
      for (const auto& curr_dev : devices)
        if (curr_dev.getInfo<CL_DEVICE_AVAILABLE>())
        {
          auto default_platform = cl::Platform::setDefault(curr_plat);
          if (default_platform != curr_plat)
            throw std::runtime_error("Failed to set default CL platform");
          auto default_device = cl::Device::setDefault(curr_dev);
          if (default_device != curr_dev)
            throw std::runtime_error("Failed to set default CL device");
          auto context = cl::Context(default_device);
          auto default_context = cl::Context::setDefault(context);
          if (default_context != context)
            throw std::runtime_error("Failed to set default CL context");
          auto cmd_queue = cl::CommandQueue(default_context, default_device);
          auto default_cmd_queue = cl::CommandQueue::setDefault(cmd_queue);
          if (default_cmd_queue != cmd_queue)
            throw std::runtime_error("Failed to set default CL command queue");
          // split extensions
          std::istringstream ext_iss(curr_dev.getInfo<CL_DEVICE_EXTENSIONS>());
          using iss_it = std::istream_iterator<std::string>;
          extensions_ = ext_vec_t(iss_it(ext_iss), iss_it{});
          std::sort(extensions_.begin(), extensions_.end());
          return;
        }
    }
    throw std::runtime_error("No suitable CL environment was set");
  }


  const cl_default_gpu_setter::ext_vec_t&
  cl_default_gpu_setter::get_extensions() const
  {
    return extensions_;
  }


  std::ostream& operator<<(std::ostream& os, const cl_default_gpu_setter& dd)
  {
    auto d = cl::Device::getDefault();
    os
      << std::boolalpha
      << "Default device:" << std::endl
      << " vendor=" << d.getInfo<CL_DEVICE_VENDOR>() << std::endl
      << " name=" << d.getInfo<CL_DEVICE_NAME>() << std::endl
      << " driver_ver=" << d.getInfo<CL_DRIVER_VERSION>() << std::endl
      << " device_ver=" << d.getInfo<CL_DEVICE_VERSION>() << std::endl
      << " cl_ver=" << d.getInfo<CL_DEVICE_OPENCL_C_VERSION>() << std::endl
      << " device_global_memory="
      << d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1 << 20) << "MB" << std::endl
      << " device_local_memory="
      << d.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() / (1 << 10) << "KB" << std::endl
      << " device_max_work_dim="
      << d.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << std::endl
      << " device_max_work_group_size="
      << d.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl
      << " device_little_endian="
      << (d.getInfo<CL_DEVICE_ENDIAN_LITTLE>() == CL_TRUE) << std::endl
      << " device_profiling_resolution="
      << d.getInfo<CL_DEVICE_PROFILING_TIMER_RESOLUTION>() << "ns" << std::endl
      << " extensions" << std::endl
      << "  [" << std::endl;
    for (const auto& curr_ext : dd.get_extensions())
      os << "   - " << curr_ext << std::endl;
    os << "  ]";
    return os;
  }

} // !namespace mfmat
