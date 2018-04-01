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
  target_link_libraries(${test_name} ${Boost_LIBRARIES})
  add_dependencies(tests ${test_name})
  add_test(${test_name} ${test_name})
  set_tests_properties(${test_name} PROPERTIES TIMEOUT 30)
endmacro()

build_and_run_test(compare_ops.cpp)
build_and_run_test(construction.cpp)
build_and_run_test(copy_vector.cpp)
build_and_run_test(determinant.cpp)
build_and_run_test(dot_product.cpp)
build_and_run_test(getters.cpp)
build_and_run_test(matrices_ops.cpp)
build_and_run_test(mean.cpp)
build_and_run_test(multiply.cpp)
build_and_run_test(norm.cpp)
build_and_run_test(normalize.cpp)
build_and_run_test(orthonormalization.cpp)
build_and_run_test(sequence_helpers.cpp)
build_and_run_test(scalar_ops.cpp)
build_and_run_test(trace.cpp)
build_and_run_test(transpose.cpp)
