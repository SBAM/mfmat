kernel void matrix_multiply(const ulong M, const ulong N, const ulong K,
                            const global data_t* A,
                            const global data_t* B,
                            global data_t* C)
{
  // Thread identifiers
  const size_t globalRow = get_global_id(0); // Row ID of C (0..M)
  const size_t globalCol = get_global_id(1); // Col ID of C (0..N)

  // Computes a single element (loop over K)
  data_t acc = 0.0;
  for (size_t k = 0; k < K; ++k)
    acc += A[k * M + globalRow] * B[globalCol * K + k];

  // Stores the result
  C[globalCol * M + globalRow] = acc;
}
