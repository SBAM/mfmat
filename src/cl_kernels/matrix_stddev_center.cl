kernel void matrix_stddev_center(global data_t* dat,
                                 const ulong M, /* dat's row count */
                                 const ulong N /* dat's column count */)
{
  const size_t col = get_global_id(0); // current column index
  data_t mean = 0.0;
  for (size_t i = 0; i < M; ++i)
    mean += dat[i * N + col];
  mean /= M;
  data_t stddev = 0.0;
  for (size_t i = 0; i < M; ++i)
  {
    data_t tmp_diff = dat[i * N + col] - mean;
    stddev += tmp_diff * tmp_diff;
  }
  stddev = sqrt(stddev / M);
  for (size_t i = 0; i < M; ++i)
    dat[i * N + col] = (dat[i * N + col] - mean) / stddev;
}
