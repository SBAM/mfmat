#define BOOST_TEST_MODULE mfmat

#include <vector>

#include <boost/test/unit_test.hpp>
#include <boost/test/tools/output_test_stream.hpp>

#include <mfmat/util/lz4_iostream.hpp>

BOOST_AUTO_TEST_SUITE(lz4_stream_test_suite)


namespace
{

  constexpr auto len = 10000ul;
  constexpr auto half_len = len / 2ul;

} // !namespace anonymous


BOOST_AUTO_TEST_CASE(compress_decompress_sstream)
{
  std::ostringstream orig_os;
  for (auto i = 0ul; i < len; ++i)
    orig_os << i << '\n';
  auto orig = orig_os.str();
  mfmat::util::lz4_ostream<std::stringstream> os;
  os << orig;
  auto sstream = os.release();
  BOOST_CHECK(os.eof());
  mfmat::util::lz4_istream<std::stringstream> is(std::move(sstream));
  auto res = std::string(std::istreambuf_iterator<char>(is), {});
  BOOST_CHECK_EQUAL(orig, res);
}


BOOST_AUTO_TEST_CASE(compress_decompress_binary)
{
  std::vector<std::uint64_t> orig_vec;
  orig_vec.reserve(len);
  for (auto i = 0ul; i < len; ++i)
    orig_vec.push_back(i);
  mfmat::util::lz4_ostream<std::stringstream> os;
  os.write(reinterpret_cast<char*>(orig_vec.data()),
           static_cast<std::streamsize>(orig_vec.size() *
                                        sizeof(std::uint64_t)));
  mfmat::util::lz4_istream<std::stringstream> is(os.release());
  for (auto i = 0ul; i < len; ++i)
  {
    std::uint64_t tmp;
    is.read(reinterpret_cast<char*>(&tmp),
            static_cast<std::streamsize>(sizeof(std::uint64_t)));
    BOOST_CHECK_EQUAL(orig_vec[i], tmp);
  }
}


BOOST_AUTO_TEST_CASE(decompress_and_seek)
{
  std::vector<std::uint64_t> orig_vec;
  orig_vec.reserve(len);
  for (auto i = 0ul; i < len; ++i)
    orig_vec.push_back(i);
  mfmat::util::lz4_ostream<std::stringstream> os;
  os.write(reinterpret_cast<char*>(orig_vec.data()),
           static_cast<std::streamsize>(orig_vec.size() *
                                        sizeof(std::uint64_t)));
  mfmat::util::lz4_istream<std::stringstream> is(os.release());
  // check first half
  for (auto i = 0ul; i < half_len; ++i)
  {
    std::uint64_t tmp;
    is.read(reinterpret_cast<char*>(&tmp),
            static_cast<std::streamsize>(sizeof(std::uint64_t)));
    BOOST_CHECK_EQUAL(orig_vec[i], tmp);
  }
  auto seek_count = 100ul;
  auto seek_len = static_cast<std::streamsize>(seek_count) *
    static_cast<std::streamsize>(sizeof(std::uint64_t));
  // rewind with invalid direction
  is.seekg(-seek_len, std::ios::beg);
  BOOST_CHECK(is.fail());
  is.clear();
  // rewind too far back
  is.seekg(-10 * seek_len, std::ios::cur);
  BOOST_CHECK(is.fail());
  is.clear();
  // fast-forward too far ahead
  is.seekg(10 * seek_len, std::ios::cur);
  BOOST_CHECK(is.fail());
  is.clear();
  // rewind to acceptable position
  is.seekg(-seek_len, std::ios::cur);
  BOOST_CHECK(!is.fail());
  // check from rewinded position to last
  for (auto i = half_len - seek_count; i < len; ++i)
  {
    std::uint64_t tmp;
    is.read(reinterpret_cast<char*>(&tmp), sizeof(std::uint64_t));
    BOOST_CHECK_EQUAL(orig_vec[i], tmp);
  }
}


BOOST_AUTO_TEST_CASE(move_constructor_test)
{
  std::vector<std::uint64_t> orig_vec;
  orig_vec.reserve(len);
  for (auto i = 0ul; i < len; ++i)
    orig_vec.push_back(i);
  // write first half using first ostream
  mfmat::util::lz4_ostream<std::stringstream> os1;
  auto* dat_ptr = orig_vec.data();
  os1.write(reinterpret_cast<char*>(dat_ptr),
            half_len * sizeof(std::uint64_t));
  // write second half using move constructed ostream
  mfmat::util::lz4_ostream<std::stringstream> os2(std::move(os1));
  dat_ptr += half_len;
  os2.write(reinterpret_cast<char*>(dat_ptr),
            half_len * sizeof(std::uint64_t));
  // check first half using first istream
  mfmat::util::lz4_istream<std::stringstream> is1(os2.release());
  for (auto i = 0ul; i < half_len; ++i)
  {
    std::uint64_t tmp;
    is1.read(reinterpret_cast<char*>(&tmp), sizeof(std::uint64_t));
    BOOST_CHECK_EQUAL(orig_vec[i], tmp);
  }
  // check second half using move constructed istream
  mfmat::util::lz4_istream<std::stringstream> is2(std::move(is1));
  for (auto i = half_len; i < len; ++i)
  {
    std::uint64_t tmp;
    is2.read(reinterpret_cast<char*>(&tmp), sizeof(std::uint64_t));
    BOOST_CHECK_EQUAL(orig_vec[i], tmp);
  }
}

BOOST_AUTO_TEST_SUITE_END()
