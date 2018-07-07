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

} // !namespace mfmat
