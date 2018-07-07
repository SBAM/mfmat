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

} // !namespace mfmat
