kernel void matrix_is_equal(const global data_t* lhs,
                            const global data_t* rhs,
                            global int* res)
{
  const size_t idx = get_global_id(0);
  if (idx == 0)
    *res = 1;
  barrier(CLK_GLOBAL_MEM_FENCE);
  const data_t abs_sum = fabs(lhs[idx]) + fabs(rhs[idx]);
  const data_t vmag = max((data_t)1.0, abs_sum);
  const int cmp = fabs(lhs[idx] - rhs[idx]) <= DATA_T_EPS * vmag;
  atomic_and(res, cmp);
}
