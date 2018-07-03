kernel void matrix_sub(global data_t* lhs, const global data_t* rhs)
{
  const size_t pos = get_global_id(0);
  lhs[pos] -= rhs[pos];
}
