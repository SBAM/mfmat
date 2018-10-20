kernel void matrix_mean_center(global data_t* dat,
                               const ulong M, /* dat's row count */
                               const ulong N /* dat's column count */)
{
  const size_t base_row = get_local_id(0); // base row index
  const size_t col = get_global_id(1); // current column index
  const size_t wg_size = get_local_size(0); // work-group size
  local data_t mean;
  if (base_row == 0)
  {
    mean = 0.0;
    for (size_t i = 0; i < M; ++i)
      mean += dat[i * N + col];
    mean /= M;
  }
  barrier(CLK_LOCAL_MEM_FENCE); // ensure mean is computed
  for (size_t row = base_row; row < M; row += wg_size)
    dat[row * N + col] -= mean;
}
