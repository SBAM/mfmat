# override default archiving tools
set(CMAKE_AR_NAME gcc-ar)
set(CMAKE_NM_NAME gcc-nm)
set(CMAKE_RANLIB_NAME gcc-ranlib)

# override default cmake flags
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb3 -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -O3")

# regroups compilation flags used to build mfmat
set(MFMAT_CXX_FLAGS
  -Wall -Wextra -Werror
  -Wconversion
  -Wdisabled-optimization
  -Wdouble-promotion
  -Wfloat-equal
  -Winline
  -Wshadow)
