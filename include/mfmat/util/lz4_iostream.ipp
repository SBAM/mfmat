namespace mfmat::util
{

  template <typename STD_T, typename BUF_T>
  lz4_iostream_t<STD_T, BUF_T>::lz4_iostream_t(STD_T&& std_s) :
    stream_(std::make_unique<STD_T>(std::move(std_s))),
    stream_buf_(std::make_unique<BUF_T>(*stream_))
  {
    this->init(stream_buf_.get());
  }


  template <typename STD_T, typename BUF_T>
  lz4_iostream_t<STD_T, BUF_T>::lz4_iostream_t(stream_uptr std_s) :
    stream_(std::move(std_s)),
    stream_buf_(std::make_unique<BUF_T>(*stream_))
  {
    this->init(stream_buf_.get());
  }


  template <typename STD_T, typename BUF_T>
  template <typename... Args>
  lz4_iostream_t<STD_T, BUF_T>::lz4_iostream_t(Args&&... args) :
    stream_(std::make_unique<STD_T>(std::forward<Args>(args)...)),
    stream_buf_(std::make_unique<BUF_T>(*stream_))
  {
    this->init(stream_buf_.get());
  }


  template <typename STD_T, typename BUF_T>
  lz4_iostream_t<STD_T, BUF_T>::lz4_iostream_t(lz4_iostream_t&& rhs) :
    STD_T(std::move(rhs)),
    stream_(std::move(rhs.stream_)),
    stream_buf_(std::move(rhs.stream_buf_))
  {
    this->set_rdbuf(stream_buf_.get());
  }


  template <typename STD_T, typename BUF_T>
  STD_T& lz4_iostream_t<STD_T, BUF_T>::get()
  {
    return *stream_;
  }


  template <typename STD_T, typename BUF_T>
  auto lz4_iostream_t<STD_T, BUF_T>::release() -> stream_uptr
  {
    this->setstate(std::ios_base::eofbit);
    stream_buf_.reset();
    return std::move(stream_);
  }

} // !namespace mfmat::util
