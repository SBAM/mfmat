# CMP0083 PIE policy introduced in 3.14
cmake_minimum_required(VERSION 3.14)

project(mfmat
  VERSION 1.0
  LANGUAGES CXX)

message("-- cmake toolchain file = ${CMAKE_TOOLCHAIN_FILE}")

# Expose LTO option
option(LTO "Enables Link-Time-Optimization" ON)

# check position independent support (-fPIE)
include(CheckPIESupported)
check_pie_supported(OUTPUT_VARIABLE PIE_MSG
  LANGUAGES CXX)
if (NOT CMAKE_CXX_LINK_PIE_SUPPORTED)
  message(FATAL_ERROR "Linker with PIE capabilities required: ${PIE_MSG}")
endif()

# check link-time-optimisation support (-flto)
if (LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT IPO_SUPPORTED
    OUTPUT IPO_MSG
    LANGUAGES CXX)
  if (NOT IPO_SUPPORTED)
    message(WARNING "LTO requested but not supported: ${IPO_MSG}")
    set(LTO OFF CACHE BOOL FORCE)
  else()
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
  endif()
endif()

# Searches for dependencies
# - pkg-config
find_package(PkgConfig REQUIRED)
# - Boost
find_package(Boost REQUIRED COMPONENTS
  unit_test_framework)
# - libLZ4
pkg_search_module(LibLZ4 REQUIRED IMPORTED_TARGET
  liblz4)
# - OpenCL
find_package(OpenCL REQUIRED)

# Provide project directory
add_subdirectory(src)

# Provide tests directory
enable_testing()
add_subdirectory(unit-tests)
