cmake_minimum_required(VERSION 3.18...4.0)

project(
  ZynNova
  VERSION 0.2.0
  DESCRIPTION "Materials intelligence, simulation, generative geometry, meshing, and speech framework"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(PYBIND11_FINDPYTHON ON)

option(ZYNNOVA_BUILD_NATIVE "Build the native C++ backends" ON)
option(ZYNNOVA_BUILD_CPP_TESTS "Build native C++ tests" OFF)
option(
  ZYNNOVA_BUILD_ZIVAR_KOKKOS
  "Build the experimental Ziver matrix-free Kokkos numerical core"
  OFF
)

if(ZYNNOVA_BUILD_NATIVE)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
endif()

if(
  ZYNNOVA_BUILD_CPP_TESTS
  AND (ZYNNOVA_BUILD_NATIVE OR ZYNNOVA_BUILD_ZIVAR_KOKKOS)
)
  enable_testing()
endif()

if(ZYNNOVA_BUILD_NATIVE OR ZYNNOVA_BUILD_ZIVAR_KOKKOS)
  add_subdirectory(cpp)
endif()
