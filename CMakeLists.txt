cmake_minimum_required(VERSION 3.9)

project(mfmat)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-std=c++17 IS_CXX17_SUPPORTED)
if (NOT IS_CXX17_SUPPORTED)
  message(FATAL_ERROR "Compiler with c++17 capabilities required")
  return()
endif()

# Pushes definitions
add_definitions(
  -DBOOST_NO_AUTO_PTR
  -DBOOST_TEST_DYN_LINK)

# Handles debug/release differences
set(CMAKE_CXX_FLAGS_DEBUG "-O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Searches for dependencies
# - Boost
set(Boost_USE_MULTITHREADED ON)
find_package(Boost REQUIRED COMPONENTS
  unit_test_framework)
# - OpenCL
find_package(OpenCL REQUIRED)
# - Threads
find_package(Threads REQUIRED)

# - Compile Flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -ggdb3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Wextra -Werror")

# Sets up global system includes directories
include_directories(SYSTEM
  ${Boost_INCLUDE_DIRS}
  ${OpenCL_INCLUDE_DIRS})

# Provide project directory
add_subdirectory(src)

# Provide tests directory
enable_testing()
add_subdirectory(unit-tests)
