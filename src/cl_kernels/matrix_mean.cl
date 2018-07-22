kernel void matrix_mean(const global data_t* arg,
                        const ulong M, /* row count */
                        const ulong N, /* column count */
                        global data_t* res)
{
  const size_t col = get_global_id(0); // res' column index
  data_t mean = 0.0;
  for (size_t i = 0; i < M; ++i)
    mean += arg[i * N + col];
  res[col] = mean / M;
}
