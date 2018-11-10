kernel void matrix_is_symmetric(const global data_t* dat,
                                const ulong N, /* dat's column count */
                                global int* res)
{
  const size_t row = get_global_id(0); // row index
  const size_t col = get_global_id(1); // column index
  // isequal function returns a 0 if the specified relation is false and a 1
  // if the specified relation is true for scalar argument types
  if (row == 0 && col == 0)
    *res = 1;
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (col > row)
  {
    const int cmp = isequal(dat[row * N + col], dat[col * N + row]);
    atomic_min(res, cmp);
  }
}
