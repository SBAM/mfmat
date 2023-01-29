include(${CMAKE_CURRENT_LIST_DIR}/CommonToolchain.cmake)

# regroups compilation flags used to build mfmat
set(MFMAT_CXX_FLAGS
  -Wall -Wextra -Werror
  -Wconversion
  -Wdisabled-optimization
  -Wdouble-promotion
  -Wfloat-equal
  -Wshadow)
