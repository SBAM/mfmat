cmake_minimum_required(VERSION 3.25.1)

project(mfmat
  VERSION 1.1
  LANGUAGES CXX)

# expose build options
option(SHARED "Shared link build" OFF)
option(LTO "Enables Link-Time-Optimization" OFF)

# check link-time-optimisation support (-flto)
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED
  OUTPUT IPO_MSG
  LANGUAGES CXX)
if (NOT IPO_SUPPORTED)
  message(WARNING "LTO not supported: ${IPO_MSG}")
endif()

# Setup conan dependencies
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(KEEP_RPATHS)

message(STATUS "cmake toolchain file = ${CMAKE_TOOLCHAIN_FILE}")
message(STATUS "Option SHARED is ${SHARED}")
message(STATUS "Option LTO is ${LTO}")

if (IPO_SUPPORTED AND LTO)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

# Searches for dependencies
# - Boost
set(Boost_NO_WARN_NEW_VERSIONS ON)
find_package(Boost REQUIRED COMPONENTS
  unit_test_framework)
# - lz4
find_package(lz4 REQUIRED)
# - OpenCL
find_package(OpenCL REQUIRED)

if (NOT SHARED)
  # Handles link flags when shared build option is OFF
  string(CONCAT CMAKE_EXE_LINKER_FLAGS
    "-static-libgcc "
    "-static-libstdc++")
  # Adjust RPATH, relative to executable/library location
  set(CMAKE_INSTALL_RPATH "$ORIGIN/")
  set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
endif()

# Provide project directory
add_subdirectory(src)

# Provide tests directory
enable_testing()
add_subdirectory(unit-tests)
