kernel void matrix_is_equal(const global data_t* lhs,
                            const global data_t* rhs,
                            global int* res)
{
  const size_t idx = get_global_id(0);
  // isequal function returns a 0 if the specified relation is false and a 1
  // if the specified relation is true for scalar argument types
  if (idx == 0)
    *res = 1;
  barrier(CLK_GLOBAL_MEM_FENCE);
  const int cmp = isequal(lhs[idx], rhs[idx]);
  atomic_min(res, cmp);
}
