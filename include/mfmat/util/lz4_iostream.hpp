#ifndef MFMAT_LZ4_IOSTREAM_HPP_
# define MFMAT_LZ4_IOSTREAM_HPP_

# include <iostream>
# include <memory>

# include "lz4_streambuf.hpp"

namespace mfmat::util
{

  /**
   * @brief Decorates a standard iostream with LZ4 (de)compression capabilities.
   * @tparam STD_T parent standard iostream
   * @tparam BUF_F specialized LZ4 streambuf
   */
  template <typename STD_T, typename BUF_T>
  class lz4_iostream_t : public STD_T
  {
  public:
    lz4_iostream_t(STD_T& std_s);
    lz4_iostream_t(lz4_iostream_t&& rhs);
    ~lz4_iostream_t() = default;

    lz4_iostream_t(const lz4_iostream_t&) = delete;
    lz4_iostream_t& operator=(const lz4_iostream_t&) = delete;

  private:
    std::unique_ptr<BUF_T> stream_buf_; ///< internal LZ4 streambuf
  };

  /// @typedef lz4_ostream shorthand to LZ4 ostream
  using lz4_ostream = lz4_iostream_t<std::ostream, lz4_ostreambuf>;
  /// @typedef lz4_istream shorthand to LZ4 istream
  using lz4_istream = lz4_iostream_t<std::istream, lz4_istreambuf>;

} // !namespace mfmat::util

# include "lz4_iostream.ipp"

#endif // !MFMAT_LZ4_IOSTREAM_HPP_
