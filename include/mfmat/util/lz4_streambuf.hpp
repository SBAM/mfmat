#ifndef MFMAT_LZ4_STREAMBUF_HPP_
# define MFMAT_LZ4_STREAMBUF_HPP_

# include <array>
# include <streambuf>
# include <vector>

# include <lz4frame.h>

namespace mfmat::util
{

  constexpr std::size_t buf_len = 4096; ///< internal buffer length

  /**
   * @brief Overrides default streambuf, decompresses an LZ4 frames conformant
   *        stream.
   *        Only basic functionalities are supported, streambuf is not
   *        seekable.
   */
  class lz4_ostreambuf : public std::streambuf
  {
  public:
    lz4_ostreambuf(std::ostream& os);
    ~lz4_ostreambuf();

  protected:
    int_type overflow(int_type ch) override;
    int_type sync() override;

  private:
    std::ostream& os_; ///< parent ostream
    std::array<char, buf_len> src_; ///< uncompressed data buffer
    std::vector<char> dest_; ///< compressed data buffer
    LZ4F_compressionContext_t ctx_; ///< LZ4 context
  };


  /**
   * @brief Overrides default streambuf, compresses a stream using LZ4 framing.
   *        This streambuf is slightly seekable, as long as requested position
   *        remains within decompressed window (defined by buf_len).
   */
  class lz4_istreambuf : public std::streambuf
  {
  public:
    lz4_istreambuf(std::istream& is);
    ~lz4_istreambuf();

  protected:
    int_type underflow() override;
    pos_type seekoff(off_type requested_offset,
                     std::ios_base::seekdir direction,
                     std::ios_base::openmode) override;

  private:
    std::istream& is_; ///< parent istream
    std::array<char, buf_len> src_; ///< compressed data buffer
    std::array<char, buf_len> dest_; ///< uncompressed data buffer
    std::size_t offset_; ///< current position in src_
    std::size_t src_buf_size_; ///< current src_ length
    LZ4F_decompressionContext_t ctx_; ///< LZ4 context
  };

} // !namespace mfmat::util

#endif // !MFMAT_LZ4_STREAMBUF_HPP_
