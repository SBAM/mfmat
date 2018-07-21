kernel void matrix_mean_center(global data_t* dat,
                               const ulong M, /* dat's row count */
                               const ulong N /* dat's column count */)
{
  const size_t col = get_global_id(0); // res' column index
  data_t acc = 0.0;
  for (size_t i = 0; i < M; ++i)
    acc += dat[i * N + col];
  acc /= M;
  for (size_t i = 0; i < M; ++i)
    dat[i * N + col] -= acc;
}
