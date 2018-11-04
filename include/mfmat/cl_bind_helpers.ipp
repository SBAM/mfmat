namespace mfmat
{

  template <typename... Ts>
  inline cl::Buffer rw_bind(std::vector<Ts...>& vec)
  {
    return cl::Buffer(vec.begin(), vec.end(),
                      false, // read/write
                      true); // contiguous data
  }


  template <typename... Ts>
  inline cl::Buffer ro_bind(const std::vector<Ts...>& vec)
  {
    /// @note works around OpenCL's const wrongness
    auto& nc_vec = const_cast<std::vector<Ts...>&>(vec);
    return cl::Buffer(nc_vec.begin(), nc_vec.end(),
                      true, // read only
                      true); // contiguous data
  }


  inline cl::Buffer wo_bind(std::size_t len)
  {
    return cl::Buffer(CL_MEM_WRITE_ONLY,
                      len, // bytes count
                      nullptr,
                      nullptr);
  }


  template <typename... Ts>
  inline cl::Buffer wo_bind(const std::vector<Ts...>& vec)
  {
    using value_type = typename std::vector<Ts...>::value_type;
    return cl::Buffer(CL_MEM_WRITE_ONLY,
                      vec.size() * sizeof(value_type), // bytes count
                      nullptr,
                      nullptr);
  }


  inline cl::Buffer no_host_bind(std::size_t len)
  {
    return cl::Buffer(CL_MEM_HOST_NO_ACCESS,
                      len, // bytes count
                      nullptr,
                      nullptr);
  }


  template <typename T, typename K, typename... Args>
  inline void bind_ker(K& ker, Args&&... ker_args)
  {
    static_assert(std::is_floating_point_v<T>);
    if constexpr (std::is_same_v<float, T>)
      ker.f.func_(std::forward<Args>(ker_args)...);
    else
      ker.d.func_(std::forward<Args>(ker_args)...);
  }


  template <typename... Ts>
  inline cl::Event bind_res(const cl::Buffer& clb, std::vector<Ts...>& vec)
  {
    using value_type = typename std::vector<Ts...>::value_type;
    auto queue = cl::CommandQueue::getDefault();
    cl::Event map_event;
    auto* ptr = static_cast<value_type*>
      (queue.enqueueMapBuffer(clb,
                              true,
                              CL_MAP_WRITE,
                              0, // offset
                              vec.size() * sizeof(value_type),
                              nullptr,
                              &map_event,
                              nullptr));
    vec.assign(ptr, ptr + vec.size());
    cl::Event unmap_event;
    queue.enqueueUnmapMemObject(clb, ptr, nullptr, &unmap_event);
    unmap_event.wait();
    return map_event;
  }


  template <typename T>
  inline cl::Event bind_res
  (const cl::Buffer& clb, T* ptr, std::size_t elt_count)
  {
    auto queue = cl::CommandQueue::getDefault();
    cl::Event event;
    queue.enqueueReadBuffer(clb,
                            true, // blocking
                            0, // offset
                            elt_count * sizeof(T),
                            ptr,
                            nullptr,
                            &event);
    return event;
  }

} // !namespace mfmat
