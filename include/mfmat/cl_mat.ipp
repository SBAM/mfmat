namespace mfmat
{

  template <typename T>
  std::size_t cl_mat<T>::get_row_count() const
  {
    return row_count_;
  }


  template <typename T>
  std::size_t cl_mat<T>::get_col_count() const
  {
    return col_count_;
  }


  template <typename T>
  typename cl_mat<T>::iterator cl_mat<T>::begin() noexcept
  {
    return storage_.begin();
  }


  template <typename T>
  typename cl_mat<T>::const_iterator cl_mat<T>::begin() const noexcept
  {
    return storage_.begin();
  }


  template <typename T>
  typename cl_mat<T>::const_iterator cl_mat<T>::cbegin() const noexcept
  {
    return storage_.cbegin();
  }


  template <typename T>
  typename cl_mat<T>::iterator cl_mat<T>::end() noexcept
  {
    return storage_.end();
  }


  template <typename T>
  typename cl_mat<T>::const_iterator cl_mat<T>::end() const noexcept
  {
    return storage_.end();
  }


  template <typename T>
  typename cl_mat<T>::const_iterator cl_mat<T>::cend() const noexcept
  {
    return storage_.cend();
  }


  template <typename T>
  T cl_mat<T>::operator[](indices idx) const
  {
    assert(idx.first < row_count_ && idx.second < col_count_);
    return storage_[idx.first * col_count_ + idx.second];
  }


  template <typename T>
  T cl_mat<T>::get(std::size_t row_idx, std::size_t col_idx) const
  {
    assert(row_idx < row_count_ && col_idx < col_count_);
    return storage_[row_idx * col_count_ + col_idx];
  }


  template <typename T>
  T& cl_mat<T>::operator[](indices idx)
  {
    assert(idx.first < row_count_ && idx.second < col_count_);
    return storage_[idx.first * col_count_ + idx.second];
  }


  template <typename T>
  T& cl_mat<T>::get(std::size_t row_idx, std::size_t col_idx)
  {
    assert(row_idx < row_count_ && col_idx < col_count_);
    return storage_[row_idx * col_count_ + col_idx];
  }

} // !namespace mfmat
