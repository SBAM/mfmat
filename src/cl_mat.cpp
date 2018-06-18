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
    row_count_(rows_cols),
    col_count_(rows_cols),
    storage_(rows_cols * rows_cols, 0.0)
  {
  }


  template <typename T>
  cl_mat<T>::cl_mat(std::size_t rows_cols, identity_tag) :
    row_count_(rows_cols),
    col_count_(rows_cols),
    storage_(rows_cols * rows_cols, 0.0)
  {
    for (std::size_t i = 0; i < rows_cols; ++i)
      storage_[rows_cols * i + i] = 1.0;
  }


  template class cl_mat<float>;
  template class cl_mat<double>;

} // !namespace mfmat
