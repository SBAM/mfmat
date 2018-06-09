set(HEADERS_DIR ${CMAKE_SOURCE_DIR}/include)
set(CL_KERNELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cl_kernels)

file(GLOB_RECURSE HEADERS ${HEADERS_DIR}/*.hpp ${HEADERS_DIR}/*.ipp)
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB CL_KERNELS ${CL_KERNELS_DIR}/*.cl)

include_directories(${HEADERS_DIR})

# static resources helper
macro(add_cl_kernels out_obj_list)
  set(local_obj_list)
  foreach (absolute_src_ext ${ARGN})
    file(RELATIVE_PATH curr_src_ext ${CL_KERNELS_DIR} ${absolute_src_ext})
    get_filename_component(curr_src ${curr_src_ext} NAME_WE)
    set(gen_obj "${PROJECT_BINARY_DIR}/${curr_src}.o")
    add_custom_command(OUTPUT ${gen_obj}
      COMMAND ${CMAKE_LINKER} -r -b binary ${curr_src_ext} -o ${gen_obj}
      DEPENDS ${absolute_src_ext}
      WORKING_DIRECTORY ${CL_KERNELS_DIR}
      COMMENT "Building OpenCL kernel ${curr_src}"
      VERBATIM)
    list(APPEND local_obj_list ${gen_obj})
  endforeach()
  set(${out_obj_list} "${local_obj_list}")
endmacro()

add_cl_kernels(CL_KERNELS_OBJ ${CL_KERNELS})

add_library(mfmat STATIC
  ${HEADERS}
  ${SOURCES}
  ${CL_KERNELS_OBJ})

target_link_libraries(mfmat
  ${OpenCL_LIBRARIES})
