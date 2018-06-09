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
          return;
        }
    }
    throw std::runtime_error("No suitable CL environment was set");
  }

} // !namespace mfmat
