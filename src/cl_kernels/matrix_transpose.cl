kernel void matrix_transpose(const global data_t* dat_in,
                             const ulong M, /* dat_in's row count */
                             const ulong N, /* dat_in's column count */
                             global data_t* dat_out)
{
  const size_t row = get_global_id(0);
  const size_t col = get_global_id(1);
  dat_out[col * M + row] = dat_in[row * N + col];
}
