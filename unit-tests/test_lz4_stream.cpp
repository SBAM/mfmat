#define BOOST_TEST_MODULE mfmat

#include <vector>

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

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
  std::stringstream sstream;
  {
    mfmat::util::lz4_ostream os(sstream);
    os << orig;
  }
  mfmat::util::lz4_istream is(sstream);
  auto res = std::string(std::istreambuf_iterator<char>(is), {});
  BOOST_CHECK_EQUAL(orig, res);
}


BOOST_AUTO_TEST_CASE(compress_decompress_binary)
{
  std::vector<std::int64_t> orig_vec;
  orig_vec.reserve(len);
  for (auto i = 0ul; i < len; ++i)
    orig_vec.push_back(i);
  std::stringstream sstream;
  {
    mfmat::util::lz4_ostream os(sstream);
    os.write(reinterpret_cast<char*>(orig_vec.data()),
             orig_vec.size() * sizeof(std::int64_t));
    sstream.sync();
  }
  mfmat::util::lz4_istream is(sstream);
  for (auto i = 0ul; i < len; ++i)
  {
    std::int64_t tmp;
    is.read(reinterpret_cast<char*>(&tmp), sizeof(std::int64_t));
    BOOST_CHECK_EQUAL(orig_vec[i], tmp);
  }
}


BOOST_AUTO_TEST_CASE(decompress_and_seek)
{
  std::vector<std::int64_t> orig_vec;
  orig_vec.reserve(len);
  for (auto i = 0ul; i < len; ++i)
    orig_vec.push_back(i);
  std::stringstream sstream;
  {
    mfmat::util::lz4_ostream os(sstream);
    os.write(reinterpret_cast<char*>(orig_vec.data()),
             orig_vec.size() * sizeof(std::int64_t));
  }
  mfmat::util::lz4_istream is(sstream);
  // check first half
  for (auto i = 0ul; i < half_len; ++i)
  {
    std::int64_t tmp;
    is.read(reinterpret_cast<char*>(&tmp), sizeof(std::int64_t));
    BOOST_CHECK_EQUAL(orig_vec[i], tmp);
  }
  std::size_t seek_count = 100;
  std::size_t seek_len = seek_count * sizeof(std::int64_t);
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
    std::int64_t tmp;
    is.read(reinterpret_cast<char*>(&tmp), sizeof(std::int64_t));
    BOOST_CHECK_EQUAL(orig_vec[i], tmp);
  }
}


BOOST_AUTO_TEST_CASE(move_constructor_test)
{
  std::vector<std::int64_t> orig_vec;
  orig_vec.reserve(len);
  for (auto i = 0ul; i < len; ++i)
    orig_vec.push_back(i);
  std::stringstream sstream;
  {
    // write first half using first ostream
    mfmat::util::lz4_ostream os1(sstream);
    auto* dat_ptr = orig_vec.data();
    os1.write(reinterpret_cast<char*>(dat_ptr),
              half_len * sizeof(std::int64_t));
    // write second half using move constructed ostream
    mfmat::util::lz4_ostream os2(std::move(os1));
    dat_ptr += half_len;
    os2.write(reinterpret_cast<char*>(dat_ptr),
              half_len * sizeof(std::int64_t));
  }
  {
    // check first half using first istream
    mfmat::util::lz4_istream is1(sstream);
    for (auto i = 0ul; i < half_len; ++i)
    {
      std::int64_t tmp;
      is1.read(reinterpret_cast<char*>(&tmp), sizeof(std::int64_t));
      BOOST_CHECK_EQUAL(orig_vec[i], tmp);
    }
    mfmat::util::lz4_istream is2(std::move(is1));
    for (auto i = half_len; i < len; ++i)
    {
      std::int64_t tmp;
      is2.read(reinterpret_cast<char*>(&tmp), sizeof(std::int64_t));
      BOOST_CHECK_EQUAL(orig_vec[i], tmp);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
