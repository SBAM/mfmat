# Customize cmake tests command
add_custom_target(tests
  COMMAND ${CMAKE_CTEST_COMMAND}
  --output-on-failure
  --timeout 60)

# build/run tests macro helper
macro(build_and_run_test source_file)
  get_filename_component(test_name ${source_file} NAME_WE)
  add_executable(${test_name} ${source_file})
  target_compile_options(${test_name}
    PRIVATE
      $<$<BOOL:${SHARED}>:BOOST_TEST_DYN_LINK>
      ${MFMAT_CXX_FLAGS})
  target_link_libraries(${test_name}
    PRIVATE
      Boost::unit_test_framework
      mfmat)
  add_dependencies(tests ${test_name})
  add_test(NAME ${test_name}
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${test_name})
  set_tests_properties(${test_name} PROPERTIES TIMEOUT 30)
endmacro()

file(GLOB TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
foreach (curr_test ${TEST_SOURCES})
  build_and_run_test(${curr_test})
endforeach()
