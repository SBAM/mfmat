#include <sstream>

#include <mfmat/cl_bind_helpers.hpp>
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


  template <typename T>
  cl_mat<T>& cl_mat<T>::operator+=(T val)
  {
    auto& ker = cl_kernels_store::instance().matrix_scalar_add;
    auto lhs_dat = rw_bind(storage_);
    bind_ker<T>(ker, cl::NDRange(storage_.size()), lhs_dat, val);
    bind_res(lhs_dat, storage_);
    return *this;
  }


  template <typename T>
  cl_mat<T>& cl_mat<T>::operator-=(T val)
  {
    auto& ker = cl_kernels_store::instance().matrix_scalar_sub;
    auto lhs_dat = rw_bind(storage_);
    bind_ker<T>(ker, cl::NDRange(storage_.size()), lhs_dat, val);
    bind_res(lhs_dat, storage_);
    return *this;
  }


  template <typename T>
  cl_mat<T>& cl_mat<T>::operator*=(T val)
  {
    auto& ker = cl_kernels_store::instance().matrix_scalar_mul;
    auto lhs_dat = rw_bind(storage_);
    bind_ker<T>(ker, cl::NDRange(storage_.size()), lhs_dat, val);
    bind_res(lhs_dat, storage_);
    return *this;
  }


  template <typename T>
  cl_mat<T>& cl_mat<T>::operator/=(T val)
  {
    auto& ker = cl_kernels_store::instance().matrix_scalar_div;
    auto lhs_dat = rw_bind(storage_);
    bind_ker<T>(ker, cl::NDRange(storage_.size()), lhs_dat, val);
    bind_res(lhs_dat, storage_);
    return *this;
  }


  template <typename T>
  cl_mat<T>& cl_mat<T>::operator+=(const cl_mat<T>& rhs)
  {
    auto lhs_dat = rw_bind(storage_);
    auto rhs_dat = ro_bind(rhs.storage_);
    if (row_count_ == rhs.row_count_ && col_count_ == rhs.col_count_)
    {
      auto& ker = cl_kernels_store::instance().matrix_add;
      bind_ker<T>(ker, cl::NDRange(storage_.size()), lhs_dat, rhs_dat);
    }
    else
      if (row_count_ == rhs.row_count_ && rhs.col_count_ == 1)
      {
        auto& ker = cl_kernels_store::instance().matrix_add_column;
        bind_ker<T>(ker, cl::NDRange(row_count_, col_count_),
                    lhs_dat, col_count_, rhs_dat);
      }
      else
        if (rhs.row_count_ == 1 && col_count_ == rhs.col_count_)
        {
          auto& ker = cl_kernels_store::instance().matrix_add_row;
          bind_ker<T>(ker, cl::NDRange(row_count_, col_count_),
                      lhs_dat, col_count_, rhs_dat);
        }
        else
        {
          std::ostringstream err;
          err
            << "cl_mat::operator+= incompatible dimensions lhs["
            << row_count_ << ',' << col_count_ << "] += rhs["
            << rhs.row_count_ << ',' << rhs.col_count_ << ']';
          throw std::out_of_range(err.str());
        }
    bind_res(lhs_dat, storage_);
    return *this;
  }


  template <typename T>
  cl_mat<T>& cl_mat<T>::operator-=(const cl_mat<T>& rhs)
  {
    auto lhs_dat = rw_bind(storage_);
    auto rhs_dat = ro_bind(rhs.storage_);
    if (row_count_ == rhs.row_count_ && col_count_ == rhs.col_count_)
    {
      auto& ker = cl_kernels_store::instance().matrix_sub;
      bind_ker<T>(ker, cl::NDRange(storage_.size()), lhs_dat, rhs_dat);
    }
    else
      if (row_count_ == rhs.row_count_ && rhs.col_count_ == 1)
      {
        auto& ker = cl_kernels_store::instance().matrix_sub_column;
        bind_ker<T>(ker, cl::NDRange(row_count_, col_count_),
                    lhs_dat, col_count_, rhs_dat);
      }
      else
        if (rhs.row_count_ == 1 && col_count_ == rhs.col_count_)
        {
          auto& ker = cl_kernels_store::instance().matrix_sub_row;
          bind_ker<T>(ker, cl::NDRange(row_count_, col_count_),
                      lhs_dat, col_count_, rhs_dat);
        }
        else
        {
          std::ostringstream err;
          err
            << "cl_mat::operator-= incompatible dimensions lhs["
            << row_count_ << ',' << col_count_ << "] -= rhs["
            << rhs.row_count_ << ',' << rhs.col_count_ << ']';
          throw std::out_of_range(err.str());
        }
    bind_res(lhs_dat, storage_);
    return *this;
  }


  template class cl_mat<float>;
  template class cl_mat<double>;

} // !namespace mfmat
