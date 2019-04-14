cmake_minimum_required(VERSION 3.9)

project(mfmat
  VERSION 1.0
  LANGUAGES CXX)

# Expose LTO option
option(LTO "Enables Link-Time-Optimization" ON)

# Check IPO(LTO) support
if (LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_result
    OUTPUT ipo_output
    LANGUAGES CXX)
  if (NOT ipo_result)
    message(WARNING "LTO requested but not supported: ${ipo_output}")
    set(LTO OFF CACHE BOOL FORCE)
  else()
    # Provide ar/nm/ranlib with LTO plugin
    get_filename_component(TOOLCHAIN_PATH
      ${CMAKE_CXX_COMPILER}
      DIRECTORY)
    message("-- Deduced TOOLCHAIN_PATH=${TOOLCHAIN_PATH}")
    set(CMAKE_AR ${TOOLCHAIN_PATH}/${CMAKE_AR_NAME})
    message("-- Using ar              =${CMAKE_AR}")
    set(CMAKE_NM ${TOOLCHAIN_PATH}/${CMAKE_NM_NAME})
    message("-- Using nm              =${CMAKE_NM}")
    set(CMAKE_RANLIB ${TOOLCHAIN_PATH}/${CMAKE_RANLIB_NAME})
    message("-- Using ranlib          =${CMAKE_RANLIB}")
  endif()
endif()

# Searches for dependencies
# - pkg-config
find_package(PkgConfig REQUIRED)
# - Boost
set(Boost_USE_MULTITHREADED ON)
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
