#include <sstream>

#include <mfmat/cl_bind_helpers.hpp>
#include <mfmat/cl_mat_externals.hpp>

namespace mfmat
{

  template <typename T>
  cl_mat<T> operator+(cl_mat<T> lhs, T rhs)
  {
    lhs += rhs;
    return lhs;
  }
  template cl_mat_f operator+(cl_mat_f, float);
  template cl_mat_d operator+(cl_mat_d, double);


  template <typename T>
  cl_mat<T> operator+(T lhs, cl_mat<T> rhs)
  {
    return rhs + lhs;
  }
  template cl_mat_f operator+(float, cl_mat_f);
  template cl_mat_d operator+(double, cl_mat_d);


  template <typename T>
  cl_mat<T> operator-(cl_mat<T> lhs, T rhs)
  {
    lhs -= rhs;
    return lhs;
  }
  template cl_mat_f operator-(cl_mat_f, float);
  template cl_mat_d operator-(cl_mat_d, double);


  template <typename T>
  cl_mat<T> operator*(cl_mat<T> lhs, T rhs)
  {
    lhs *= rhs;
    return lhs;
  }
  template cl_mat_f operator*(cl_mat_f, float);
  template cl_mat_d operator*(cl_mat_d, double);


  template <typename T>
  cl_mat<T> operator*(T lhs, cl_mat<T> rhs)
  {
    return rhs * lhs;
  }
  template cl_mat_f operator*(float, cl_mat_f);
  template cl_mat_d operator*(double, cl_mat_d);


  template <typename T>
  cl_mat<T> operator/(cl_mat<T> lhs, T rhs)
  {
    lhs /= rhs;
    return lhs;
  }
  template cl_mat_f operator/(cl_mat_f, float);
  template cl_mat_d operator/(cl_mat_d, double);


  template <typename T>
  cl_mat<T> operator+(cl_mat<T> lhs, cl_mat<T> rhs)
  {
    lhs += rhs;
    return lhs;
  }
  template cl_mat_f operator+(cl_mat_f, cl_mat_f);
  template cl_mat_d operator+(cl_mat_d, cl_mat_d);


  template <typename T>
  cl_mat<T> operator-(cl_mat<T> lhs, cl_mat<T> rhs)
  {
    lhs -= rhs;
    return lhs;
  }
  template cl_mat_f operator-(cl_mat_f, cl_mat_f);
  template cl_mat_d operator-(cl_mat_d, cl_mat_d);


  template <typename T>
  cl_mat<T> transpose(const cl_mat<T>& arg)
  {
    cl_mat<T> res(arg.get_col_count(), arg.get_row_count());
    auto dat_in = ro_bind(arg.storage_);
    auto dat_out = rw_bind(res.storage_);
    auto& ker = cl_kernels_store::instance().matrix_transpose;
    bind_ker<T>(ker, cl::NDRange(arg.get_row_count(), arg.get_col_count()),
                dat_in, arg.get_row_count(), arg.get_col_count(), dat_out);
    bind_res(dat_out, res.storage_);
    return res;
  }
  template cl_mat_f transpose(const cl_mat_f&);
  template cl_mat_d transpose(const cl_mat_d&);


  template <typename T>
  cl_mat<T> operator*(const cl_mat<T>& lhs, const cl_mat<T>& rhs)
  {
    if (lhs.get_col_count() != rhs.get_row_count())
    {
      std::ostringstream err;
      err
        << "operator* incompatible dimensions lhs["
        << lhs.get_row_count() << ',' << lhs.get_col_count() << "] * rhs["
        << rhs.get_row_count() << ',' << rhs.get_col_count() << ']';
      throw std::runtime_error(err.str());
    }
    auto lhs_dat = ro_bind(lhs.storage_);
    auto rhs_dat = ro_bind(rhs.storage_);
    cl_mat<T> res(lhs.get_row_count(), rhs.get_col_count());
    auto res_dat = rw_bind(res.storage_);
    auto& ker = cl_kernels_store::instance().matrix_multiply;
    bind_ker<T>(ker, cl::NDRange(res.get_row_count(), res.get_col_count()),
                lhs_dat, lhs.get_col_count(),
                rhs_dat, rhs.get_col_count(),
                res_dat);
    bind_res(res_dat, res.storage_);
    return res;
  }
  template cl_mat_f operator*(const cl_mat_f&, const cl_mat_f&);
  template cl_mat_d operator*(const cl_mat_d&, const cl_mat_d&);


  template <typename T>
  cl_mat<T> mean(const cl_mat<T>& arg)
  {
    if (arg.get_row_count() == 0)
    {
      std::ostringstream err;
      err
        << "mean invalid matrix dimension [" << arg.get_row_count() << ','
        << arg.get_col_count() << ']';
      throw std::runtime_error(err.str());
    }
    auto arg_dat = ro_bind(arg.storage_);
    cl_mat<T> res(1, arg.get_col_count());
    auto res_dat = rw_bind(res.storage_);
    auto& ker = cl_kernels_store::instance().matrix_mean;
    bind_ker<T>(ker, cl::NDRange(res.get_col_count()),
                arg_dat, arg.get_row_count(), arg.get_col_count(),
                res_dat);
    bind_res(res_dat, res.storage_);
    return res;
  }
  template cl_mat_f mean(const cl_mat_f&);
  template cl_mat_d mean(const cl_mat_d&);

} // !namespace mfmat
