cmake_minimum_required(VERSION 3.9)

project(mfmat)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-std=c++17 IS_CXX17_SUPPORTED)
if (NOT IS_CXX17_SUPPORTED)
  message(FATAL_ERROR "Compiler with c++17 (-std=c++17) capabilities required")
  return()
endif()

option(LTO "Enables Link-Time-Optimization" ON)

# Replace archive tools
if (NOT $ENV{CMAKE_AR} STREQUAL "")
  message("-- Using ar    =$ENV{CMAKE_AR}")
  set(CMAKE_AR $ENV{CMAKE_AR})
endif()
if (NOT $ENV{CMAKE_NM} STREQUAL "")
  message("-- Using nm    =$ENV{CMAKE_NM}")
  set(CMAKE_NM $ENV{CMAKE_NM})
endif()
if (NOT $ENV{CMAKE_RANLIB} STREQUAL "")
  message("-- Using ranlib=$ENV{CMAKE_RANLIB}")
  set(CMAKE_RANLIB $ENV{CMAKE_RANLIB})
endif()

# Common compile flags
string(CONCAT CMAKE_CXX_FLAGS
  "-std=c++17 -ggdb3 -fPIC "
  "-Wall -Wextra -Werror "
  "-Wshadow "
  "-Wfloat-equal "
  "-Wdouble-promotion "
  "-Wunreachable-code")

# Handles debug/release differences-
string(CONCAT CMAKE_CXX_FLAGS_DEBUG
  "-O0")
string(CONCAT CMAKE_CXX_FLAGS_RELEASE
  "-O3 "
  "-DNDEBUG")
if (LTO)
  string(CONCAT CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} "
    "-flto")
endif()

# Push definitions
add_definitions(
  -DBOOST_TEST_DYN_LINK
  -DGIT_AUTHOR=$ENV{GIT_AUTHOR}
  -DGIT_COMMIT_DATE=$ENV{GIT_COMMIT_DATE}
  -DGIT_COMMIT_HASH=$ENV{GIT_COMMIT_HASH}
  -DGIT_TAG=$ENV{GIT_TAG})

# Searches for dependencies
# - Boost
set(Boost_USE_MULTITHREADED ON)
find_package(Boost REQUIRED COMPONENTS
  unit_test_framework)
# - OpenCL
find_package(OpenCL REQUIRED)
# - Threads
find_package(Threads REQUIRED)

# Sets up global system includes directories
include_directories(SYSTEM
  ${Boost_INCLUDE_DIRS}
  ${OpenCL_INCLUDE_DIRS})

# Provide project directory
add_subdirectory(src)

# Provide tests directory
enable_testing()
add_subdirectory(unit-tests)
