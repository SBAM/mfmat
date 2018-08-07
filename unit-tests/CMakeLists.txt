include_directories(${CMAKE_SOURCE_DIR}/include)

# Customize cmake tests command
add_custom_target(tests
  COMMAND ${CMAKE_CTEST_COMMAND}
  --output-on-failure
  --timeout 60)

# build/run tests macro helper
macro(build_and_run_test source_file)
  get_filename_component(test_name ${source_file} NAME_WE)
  add_executable(${test_name} ${source_file})
  target_link_libraries(${test_name}
    mfmat
    ${Boost_LIBRARIES}
    ${OpenCL_LIBRARIES})
  add_dependencies(tests ${test_name})
  add_test(${test_name} ${test_name})
  set_tests_properties(${test_name} PROPERTIES TIMEOUT 30)
endmacro()

build_and_run_test(cl_construction.cpp)
build_and_run_test(cl_correlation.cpp)
build_and_run_test(cl_covariance.cpp)
build_and_run_test(cl_kernels_store_initialization.cpp)
build_and_run_test(cl_matrices_ops.cpp)
build_and_run_test(cl_mean.cpp)
build_and_run_test(cl_mean_center.cpp)
build_and_run_test(cl_multiply.cpp)
build_and_run_test(cl_scalar_ops.cpp)
build_and_run_test(cl_standard_deviation.cpp)
build_and_run_test(cl_stddev_center.cpp)
build_and_run_test(cl_transpose.cpp)

build_and_run_test(ct_compare_ops.cpp)
build_and_run_test(ct_construction.cpp)
build_and_run_test(ct_copy_vector.cpp)
build_and_run_test(ct_correlation.cpp)
build_and_run_test(ct_covariance.cpp)
build_and_run_test(ct_determinant.cpp)
build_and_run_test(ct_dot_product.cpp)
build_and_run_test(ct_getters.cpp)
build_and_run_test(ct_is_diagonal.cpp)
build_and_run_test(ct_is_symmetric.cpp)
build_and_run_test(ct_matrices_ops.cpp)
build_and_run_test(ct_mean.cpp)
build_and_run_test(ct_mean_center.cpp)
build_and_run_test(ct_multiply.cpp)
build_and_run_test(ct_norm.cpp)
build_and_run_test(ct_normalize.cpp)
build_and_run_test(ct_orthonormalization.cpp)
build_and_run_test(ct_qr_decomposition.cpp)
build_and_run_test(ct_qr_eigen.cpp)
build_and_run_test(ct_sequence_helpers.cpp)
build_and_run_test(ct_scalar_ops.cpp)
build_and_run_test(ct_standard_deviation.cpp)
build_and_run_test(ct_stddev_center.cpp)
build_and_run_test(ct_trace.cpp)
build_and_run_test(ct_transpose.cpp)
