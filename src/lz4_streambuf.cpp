#include <sstream>

#include <mfmat/util/lz4_streambuf.hpp>

namespace mfmat::util
{

  lz4_ostreambuf::lz4_ostreambuf(std::ostream& os) :
    os_(os),
    dest_(LZ4F_compressBound(src_.size(), nullptr)),
    ctx_(nullptr)
  {
    auto* base = &src_.front();
    setp(base, base + src_.size() - 1);
    auto ccc_res = LZ4F_createCompressionContext(&ctx_, LZ4F_VERSION);
    if (LZ4F_isError(ccc_res) != 0)
    {
      std::ostringstream err;
      err
        << "LZ4F_createCompressionContext failure="
        << LZ4F_getErrorName(ccc_res);
      throw std::runtime_error(err.str());
    }
    auto cb_res = LZ4F_compressBegin(ctx_,
                                     &dest_.front(),
                                     dest_.capacity(),
                                     nullptr);
    if (LZ4F_isError(cb_res) != 0)
    {
      std::ostringstream err;
      err
        << "LZ4F_compressBegin failure="
        << LZ4F_getErrorName(cb_res);
      throw std::runtime_error(err.str());
    }
    os_.write(&dest_.front(),  static_cast<std::streamsize>(cb_res));
  }


  lz4_ostreambuf::~lz4_ostreambuf()
  {
    sync();
    auto ce_res = LZ4F_compressEnd(ctx_,
                                   &dest_.front(),
                                   dest_.capacity(),
                                   nullptr);
    if (!LZ4F_isError(ce_res))
      os_.write(&dest_.front(), static_cast<std::streamsize>(ce_res));
    LZ4F_freeCompressionContext(ctx_);
  }


  auto lz4_ostreambuf::overflow(int_type ch) -> int_type
  {
    *pptr() = static_cast<char_type>(ch);
    pbump(1);
    sync();
    return ch;
  }


  auto lz4_ostreambuf::sync() -> int_type
  {
    auto orig_size = static_cast<int>(pptr() - pbase());
    pbump(-orig_size);
    auto cu_res = LZ4F_compressUpdate(ctx_,
                                      &dest_.front(),
                                      dest_.capacity(),
                                      pbase(),
                                      static_cast<std::size_t>(orig_size),
                                      nullptr);
    if (LZ4F_isError(cu_res) != 0)
    {
      std::ostringstream err;
      err
        << "LZ4F_compressUpdate failure="
        << LZ4F_getErrorName(cu_res);
      throw std::runtime_error(err.str());
    }
    os_.write(&dest_.front(), static_cast<std::streamsize>(cu_res));
    return 0;
  }



  lz4_istreambuf::lz4_istreambuf(std::istream& is) :
    is_(is),
    offset_{},
    src_buf_size_{},
    ctx_(nullptr)
  {
    auto cdc_res = LZ4F_createDecompressionContext(&ctx_, LZ4F_VERSION);
    if (LZ4F_isError(cdc_res) != 0)
    {
      std::ostringstream err;
      err
        << "LZ4F_createDecompressionContext failure="
        << LZ4F_getErrorName(cdc_res);
      throw std::runtime_error(err.str());
    }
    setg(&src_.front(), &src_.front(), &src_.front());
  }


  lz4_istreambuf::~lz4_istreambuf()
  {
    LZ4F_freeDecompressionContext(ctx_);
  }


  auto lz4_istreambuf::underflow() -> int_type
  {
    std::size_t written_size{};
    while (written_size == 0)
    {
      if (offset_ == src_buf_size_)
      {
        is_.read(&src_.front(), static_cast<std::streamsize>(src_.size()));
        src_buf_size_ = static_cast<std::size_t>(is_.gcount());
        offset_ = 0;
      }
      if (src_buf_size_ == 0)
        return traits_type::eof();
      auto src_size = src_buf_size_ - offset_;
      auto dest_size = dest_.size();
      auto d_res = LZ4F_decompress(ctx_,
                                   &dest_.front(),
                                   &dest_size,
                                   &src_.front() + offset_,
                                   &src_size,
                                   nullptr);
      if (LZ4F_isError(d_res) != 0)
      {
        std::ostringstream err;
        err
          << "LZ4F_decompress failure="
          << LZ4F_getErrorName(d_res);
        throw std::runtime_error(err.str());
      }
      written_size = dest_size;
      offset_ += src_size;
    }
    setg(&dest_.front(), &dest_.front(), &dest_.front() + written_size);
    return *gptr();
  }


  auto lz4_istreambuf::seekoff(off_type requested_offset,
                               std::ios_base::seekdir direction,
                               std::ios_base::openmode) -> pos_type
  {
    // only seeking from current position is allowed
    if (direction != std::ios_base::cur)
      return -1;
    auto* start_ptr = eback();
    auto* curr_ptr = gptr();
    auto* end_ptr = egptr();
    auto* req_ptr = curr_ptr + requested_offset;
    // prevent seeking too far back
    if (req_ptr < start_ptr)
      return -1;
    // prevent seeking too far ahead
    if (req_ptr > end_ptr)
      return -1;
    setg(&dest_.front(),
         &dest_.front() + (req_ptr - start_ptr),
         &dest_.front() + (end_ptr - start_ptr));
    return gptr() - eback();
  }

} // !namespace mfmat::util
