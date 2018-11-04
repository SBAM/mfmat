kernel void matrix_transpose_multiply
(const global data_t* lhs,
 const ulong lhs_M, /* lhs's row count */
 const global data_t* rhs,
 const ulong rhs_N, /* rhs's column count */
 global data_t* res)
{
  const size_t row = get_global_id(0); // res' row index
  const size_t col = get_global_id(1); // res' column index
  // dot product of lhs' column with rhs' column
  data_t acc = 0.0;
  for (size_t k = 0; k < lhs_M; ++k)
    acc += lhs[k * rhs_N + row] * rhs[k * rhs_N + col];
  // store result
  res[row * rhs_N + col] = acc;
}
