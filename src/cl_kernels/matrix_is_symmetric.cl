kernel void matrix_is_symmetric(const global data_t* dat,
                                const ulong N, /* dat's column count */
                                global int* res)
{
  const size_t row = get_global_id(0); // row index
  const size_t col = get_global_id(1); // column index
  if (row == 0 && col == 0)
    *res = 1;
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (col > row)
  {
    const data_t lhs = dat[row * N + col];
    const data_t rhs = dat[col * N + row];
    const data_t vmag = max((data_t)1.0, fabs(lhs) + fabs(rhs));
    const int cmp = fabs(lhs - rhs) <= DATA_T_EPS * vmag;
    atomic_and(res, cmp);
  }
}
