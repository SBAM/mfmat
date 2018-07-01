kernel void matrix_scalar_add(global data_t* lhs, const data_t rhs)
{
  const size_t pos = get_global_id(0);
  lhs[pos] += rhs;
}
