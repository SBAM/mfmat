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
    /// @typedef stream_uptr shorthand to parent stream unique_ptr
    using stream_uptr = std::unique_ptr<STD_T>;

  public:
    /// @brief Moves std_s into internal stream unique_ptr
    lz4_iostream_t(STD_T&& std_s);
    // @brief Directly stores given stream unique_ptr
    lz4_iostream_t(stream_uptr std_s);
    // @brief Constructs internal stream unique_ptr using Args
    template <typename... Args>
    lz4_iostream_t(Args&&... args);
    // @brief Move constructs from given lz4_iostream
    lz4_iostream_t(lz4_iostream_t&& rhs);
    ~lz4_iostream_t() = default;

    lz4_iostream_t(const lz4_iostream_t&) = delete;
    lz4_iostream_t& operator=(const lz4_iostream_t&) = delete;

    /// @return reference to underlying parent stream
    STD_T& get();
    /**
     * @return internal stream unique_ptr
     * @note Calling release makes this stream no longer usable, eofbit is set.
     */
    stream_uptr release();

  private:
    stream_uptr stream_; ///< parent stream
    std::unique_ptr<BUF_T> stream_buf_; ///< internal LZ4 streambuf
  };


  /// @typedef lz4_ostream shorthand to LZ4 ostream
  template <typename OS_T>
  using lz4_ostream = lz4_iostream_t<OS_T, lz4_ostreambuf>;

 /// @typedef lz4_istream shorthand to LZ4 istream
  template <typename IS_T>
  using lz4_istream = lz4_iostream_t<IS_T, lz4_istreambuf>;

} // !namespace mfmat::util

# include "lz4_iostream.ipp"

#endif // !MFMAT_LZ4_IOSTREAM_HPP_
