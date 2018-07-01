#include <mfmat/cl_mat.hpp>

namespace mfmat
{


  template <typename T>
  cl_mat<T>::cl_mat(std::size_t rows, std::size_t cols) :
    row_count_(rows),
    col_count_(cols),
    storage_(rows * cols, 0.0)
  {
  }


  template <typename T>
  cl_mat<T>::cl_mat(std::size_t rows_cols) :
    cl_mat(rows_cols, rows_cols)
  {
  }


  template <typename T>
  cl_mat<T>::cl_mat(std::size_t rows_cols, identity_tag) :
    cl_mat(rows_cols, rows_cols)
  {
    for (std::size_t i = 0; i < rows_cols; ++i)
      storage_[rows_cols * i + i] = 1.0;
  }


  template <typename T>
  cl_mat<T>::cl_mat(cl_mat&& rhs) :
    row_count_(rhs.row_count_),
    col_count_(rhs.col_count_),
    storage_(std::move(rhs.storage_))
  {
    rhs.row_count_ = 0;
    rhs.col_count_ = 0;
  }


  template <typename T>
  cl_mat<T>& cl_mat<T>::operator=(cl_mat&& rhs)
  {
    row_count_ = rhs.row_count_;
    rhs.row_count_ = 0;
    col_count_ = rhs.col_count_;
    rhs.col_count_ = 0;
    storage_ = std::move(rhs.storage_);
    return *this;
  }


  template <typename K, typename T>
  void invoke_scalar_kernel(std::vector<T>& storage, T val, K& ker)
  {
    cl::Buffer dat(storage.begin(), storage.end(),
                   false, // read/write
                   true); // contiguous data
    if constexpr(std::is_same_v<float, T>)
      ker.f.func_(cl::NDRange(storage.size()), dat, val);
    else
      ker.d.func_(cl::NDRange(storage.size()), dat, val);
    auto queue = cl::CommandQueue::getDefault();
    queue.enqueueReadBuffer
      (dat, true /* blocking */, 0 /* offset */,
       storage.size() * sizeof(T),
       storage.data());
  }


  template <typename T>
  cl_mat<T>& cl_mat<T>::operator+=(T val)
  {
    auto& ker = cl_kernels_store::instance().matrix_scalar_add;
    invoke_scalar_kernel(storage_, val, ker);
    return *this;
  }


  template <typename T>
  cl_mat<T>& cl_mat<T>::operator-=(T val)
  {
    auto& ker = cl_kernels_store::instance().matrix_scalar_sub;
    invoke_scalar_kernel(storage_, val, ker);
    return *this;
  }


  template <typename T>
  cl_mat<T>& cl_mat<T>::operator*=(T val)
  {
    auto& ker = cl_kernels_store::instance().matrix_scalar_mul;
    invoke_scalar_kernel(storage_, val, ker);
    return *this;
  }


  template <typename T>
  cl_mat<T>& cl_mat<T>::operator/=(T val)
  {
    auto& ker = cl_kernels_store::instance().matrix_scalar_div;
    invoke_scalar_kernel(storage_, val, ker);
    return *this;
  }

  template class cl_mat<float>;
  template class cl_mat<double>;

} // !namespace mfmat
