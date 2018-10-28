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


  template <typename T>
  cl_mat<T>& cl_mat<T>::orthonormalize()
  {
    if (row_count_ == 0)
    {
      std::ostringstream err;
      err
        << "cl_mat::orthonormalize invalid matrix dimension ["
        << row_count_ << ',' << col_count_ << ']';
      throw std::runtime_error(err.str());
    }
    auto dat = rw_bind(storage_);
    auto mwgs = cl_default_gpu_setter::instance().get_max_work_group_size();
    auto& ker = cl_kernels_store::instance().matrix_orthonormalize;
    auto range = std::min(row_count_, mwgs);
    auto range_args = cl::EnqueueArgs
      {
        cl::NDRange(range),
        cl::NDRange(range)
      };
    bind_ker<T>(ker, range_args, dat, row_count_, col_count_,
                cl::Local(range * sizeof(T)));
    bind_res(dat, storage_);
    return *this;
  }


  template <typename T>
  cl_mat<T>& cl_mat<T>::transpose()
  {
    if (row_count_ == col_count_)
    {
      auto dat = rw_bind(storage_);
      auto& ker = cl_kernels_store::instance().matrix_square_transpose;
      bind_ker<T>(ker, cl::NDRange(row_count_, col_count_),
                  dat, row_count_, col_count_);
      bind_res(dat, storage_);
    }
    else
    {
      auto dat_in = ro_bind(storage_);
      storage_t out_storage(storage_.size());
      auto dat_out = wo_bind(out_storage);
      auto& ker = cl_kernels_store::instance().matrix_transpose;
      bind_ker<T>(ker, cl::NDRange(row_count_, col_count_),
                  dat_in, row_count_, col_count_, dat_out);
      bind_res(dat_out, out_storage);
      std::swap(row_count_, col_count_);
      std::swap(storage_, out_storage);
    }
    return *this;
  }


  template <typename T>
  cl_mat<T>& cl_mat<T>::mean_center()
  {
    if (row_count_ == 0)
    {
      std::ostringstream err;
      err
        << "cl_mat::mean_center invalid matrix dimension ["
        << row_count_ << ',' << col_count_ << ']';
      throw std::runtime_error(err.str());
    }
    auto dat = rw_bind(storage_);
    auto mwgs = cl_default_gpu_setter::instance().get_max_work_group_size();
    auto& ker = cl_kernels_store::instance().matrix_mean_center;
    auto range_args = cl::EnqueueArgs
      {
        cl::NDRange(std::min(row_count_, mwgs), col_count_),
        cl::NDRange(std::min(row_count_, mwgs), 1)
      };
    bind_ker<T>(ker, range_args, dat, row_count_, col_count_);
    bind_res(dat, storage_);
    return *this;
  }


  template <typename T>
  cl_mat<T>& cl_mat<T>::stddev_center()
  {
    if (row_count_ == 0)
    {
      std::ostringstream err;
      err
        << "cl_mat::stddev_center invalid matrix dimension ["
        << row_count_ << ',' << col_count_ << ']';
      throw std::runtime_error(err.str());
    }
    auto dat = rw_bind(storage_);
    auto mwgs = cl_default_gpu_setter::instance().get_max_work_group_size();
    auto& ker = cl_kernels_store::instance().matrix_stddev_center;
    auto range_args = cl::EnqueueArgs
      {
        cl::NDRange(std::min(row_count_, mwgs), col_count_),
        cl::NDRange(std::min(row_count_, mwgs), 1)
      };
    bind_ker<T>(ker, range_args, dat, row_count_, col_count_);
    bind_res(dat, storage_);
    return *this;
  }


  template class cl_mat<float>;
  template class cl_mat<double>;

} // !namespace mfmat
