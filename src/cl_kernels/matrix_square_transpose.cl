kernel void matrix_square_transpose(global data_t* dat,
                                    const ulong M, /* dat's row count */
                                    const ulong N /* dat's column count */)
{
  const size_t row = get_global_id(0);
  const size_t col = get_global_id(1);
  if (col > row)
  {
    const size_t idx1 = row * N + col;
    const size_t idx2 = col * M + row;
    const data_t tmp = dat[idx1];
    dat[idx1] = dat[idx2];
    dat[idx2] = tmp;
  }
}
