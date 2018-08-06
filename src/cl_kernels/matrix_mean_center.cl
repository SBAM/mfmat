kernel void matrix_mean_center(global data_t* dat,
                               const ulong M, /* dat's row count */
                               const ulong N /* dat's column count */)
{
  const size_t col = get_group_id(1); // current column index
  local data_t mean;
  mean = 0.0;
  for (size_t i = 0; i < M; ++i)
    mean += dat[i * N + col];
  mean /= M;
  barrier(CLK_LOCAL_MEM_FENCE); // ensure mean is computed
  const size_t row = get_local_id(0); // current row index
  dat[row * N + col] -= mean;
}
