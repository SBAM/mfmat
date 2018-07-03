kernel void matrix_add_row(global data_t* lhs,
                           const ulong N, /* lhs's column count */
                           const global data_t* rhs)
{
  const size_t row = get_global_id(0);
  const size_t col = get_global_id(1);
  lhs[row * N + col] += rhs[col];
}
