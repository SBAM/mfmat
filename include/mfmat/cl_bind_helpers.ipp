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
  inline void bind_res(const cl::Buffer& clb, std::vector<Ts...>& vec)
  {
    using value_type = typename std::vector<Ts...>::value_type;
    auto queue = cl::CommandQueue::getDefault();
    queue.enqueueReadBuffer(clb,
                            true, // blocking
                            0, // offset
                            vec.size() * sizeof(value_type),
                            vec.data());

  }

} // !namespace mfmat
