kernel void matrix_is_diagonal(const global data_t* dat,
                               const ulong N, /* dat's column count */
                               global int* res)
{
  const size_t row = get_global_id(0); // row index
  const size_t col = get_global_id(1); // column index
  if (row == 0 && col == 0)
    *res = 1;
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (row != col)
  {
    const int cmp = fabs(dat[row * N + col]) <= DATA_T_EPS;
    atomic_and(res, cmp);
  }
}
