namespace mfmat::util
{

  template <typename STD_T, typename BUF_T>
  lz4_iostream_t<STD_T, BUF_T>::lz4_iostream_t(STD_T& std_s) :
    STD_T(std::make_unique<BUF_T>(std_s).release())
  {
    stream_buf_.reset(static_cast<BUF_T*>(this->rdbuf()));
  }


  template <typename STD_T, typename BUF_T>
  lz4_iostream_t<STD_T, BUF_T>::lz4_iostream_t(lz4_iostream_t&& rhs) :
    STD_T(std::move(rhs)),
    stream_buf_(std::move(rhs.stream_buf_))
  {
    STD_T::set_rdbuf(stream_buf_.get());
  }

} // !namespace mfmat::util
