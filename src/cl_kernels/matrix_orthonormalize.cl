kernel void matrix_orthonormalize(global data_t* dat,
                                  const ulong M, /* dat's row count */
                                  const ulong N, /* dat's column count */
                                  local data_t* loc)
{
  const size_t base_row = get_global_id(0); // base row index
  const size_t wg_size = get_local_size(0); // work-group size
  size_t min_mn = min(M, N);
  // loop over all columns
  for (size_t col = 0; col < min_mn; ++col)
  {
    // project against previous columns
    for (size_t col_proj = 0; col_proj < col; ++col_proj)
    {
      // compute dot product for later projection
      loc[base_row] = (data_t)0.0;
      for (size_t row = base_row; row < M; row += wg_size)
        loc[base_row] += dat[row * N + col] * dat[row * N + col_proj];
      barrier(CLK_LOCAL_MEM_FENCE);
      local data_t local_dot;
      if (base_row == 0)
      {
        local_dot = (data_t)0.0;
        for (size_t i = 0; i < wg_size; ++i)
          local_dot += loc[i];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      // substract projection
      for (size_t row = base_row; row < M; row += wg_size)
        dat[row * N + col] -= local_dot * dat[row * N + col_proj];
      barrier(CLK_GLOBAL_MEM_FENCE);
    }
    // normalize result colum
    loc[base_row] = (data_t)0.0;
    for (size_t row = base_row; row < M; row += wg_size)
      loc[base_row] += dat[row * N + col] * dat[row * N + col];
    barrier(CLK_LOCAL_MEM_FENCE);
    local data_t norm;
    if (base_row == 0)
    {
      norm = (data_t)0.0;
      for (size_t i = 0; i < wg_size; ++i)
        norm += loc[i];
      norm = sqrt(norm);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (norm < DATA_T_EPS)
      // zero vector is normalized as zero vector
      for (size_t row = base_row; row < M; row += wg_size)
        dat[row * N + col] = (data_t)0.0;
    else
      for (size_t row = base_row; row < M; row += wg_size)
        dat[row * N + col] /= norm;
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
  // zero out remaining linearly dependant columns
  for (size_t col = min_mn; col < N; ++col)
    for (size_t row = base_row; row < M; row += wg_size)
      dat[row * N + col] = (data_t)0.0;
}
