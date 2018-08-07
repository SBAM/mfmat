kernel void matrix_self_multiply_regularized(const global data_t* dat,
                                             // dat's row count
                                             const ulong dat_M,
                                             // dat's column count
                                             const ulong dat_N,
                                             global data_t* res)
{
  const size_t row = get_global_id(0); // res' row index
  const size_t col = get_global_id(1); // res' column index
  // 1/num_obs * transpose(dat) * dat
  if (row <= col)
  {
    data_t acc = 0.0;
    for (size_t k = 0; k < dat_M; ++k)
      acc += dat[k * dat_N + row] * dat[k * dat_N + col];
    // store result
    acc /= dat_M;
    res[row * dat_N + col] = acc;
    res[col * dat_N + row] = acc;
  }
}
