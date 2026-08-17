if(ZYNNOVA_BUILD_NATIVE)
add_library(zynnova_structure_core STATIC
  src/structure/graph_builder.cpp
)

add_library(ZynNova::structure_core ALIAS zynnova_structure_core)

target_include_directories(zynnova_structure_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(zynnova_structure_core PUBLIC cxx_std_17)
set_target_properties(zynnova_structure_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  target_compile_options(zynnova_structure_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(zynnova_structure_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

pybind11_add_module(_structure_native MODULE
  bindings/structure_module.cpp
)

target_link_libraries(_structure_native PRIVATE ZynNova::structure_core)
target_include_directories(_structure_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

install(TARGETS _structure_native
  LIBRARY DESTINATION zynnova/_native
  RUNTIME DESTINATION zynnova/_native
)

add_library(zynnova_dft_core STATIC
  src/dft/aimd.cpp
  src/dft/quantum.cpp
)

add_library(ZynNova::dft_core ALIAS zynnova_dft_core)

target_include_directories(zynnova_dft_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(zynnova_dft_core PUBLIC cxx_std_17)
set_target_properties(zynnova_dft_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  target_compile_options(zynnova_dft_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(zynnova_dft_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(zynnova_dft_core PUBLIC OpenMP::OpenMP_CXX)
endif()

pybind11_add_module(_dft_native MODULE
  bindings/dft_module.cpp
)

target_link_libraries(_dft_native PRIVATE ZynNova::dft_core)
target_include_directories(_dft_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

install(TARGETS _dft_native
  LIBRARY DESTINATION zynnova/_native
  RUNTIME DESTINATION zynnova/_native
)

if(ZYNNOVA_BUILD_CPP_TESTS)
  add_executable(zynnova_dft_core_tests
    tests/test_dft_core.cpp
  )
  target_link_libraries(zynnova_dft_core_tests PRIVATE ZynNova::dft_core)
  add_test(NAME zynnova_dft_core_tests COMMAND zynnova_dft_core_tests)
endif()

add_library(zynnova_zynsim_fem_core STATIC
  src/zynsim/fem.cpp
)

add_library(ZynNova::zynsim_fem_core ALIAS zynnova_zynsim_fem_core)

target_include_directories(zynnova_zynsim_fem_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(zynnova_zynsim_fem_core PUBLIC cxx_std_17)
set_target_properties(zynnova_zynsim_fem_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  target_compile_options(zynnova_zynsim_fem_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(zynnova_zynsim_fem_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(zynnova_zynsim_fem_core PUBLIC OpenMP::OpenMP_CXX)
endif()

pybind11_add_module(_zynsim_fem_native MODULE
  bindings/zynsim_fem_module.cpp
)

target_link_libraries(_zynsim_fem_native PRIVATE ZynNova::zynsim_fem_core)
target_include_directories(_zynsim_fem_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

install(TARGETS _zynsim_fem_native
  LIBRARY DESTINATION zynnova/_native
  RUNTIME DESTINATION zynnova/_native
)

if(ZYNNOVA_BUILD_CPP_TESTS)
  add_executable(zynnova_zynsim_fem_core_tests
    tests/test_zynsim_fem.cpp
  )
  target_link_libraries(zynnova_zynsim_fem_core_tests PRIVATE ZynNova::zynsim_fem_core)
  add_test(NAME zynnova_zynsim_fem_core_tests COMMAND zynnova_zynsim_fem_core_tests)
endif()

# ZynSim high-throughput image/voxel kernels.
add_library(zynnova_zynsim_voxel_core STATIC
  src/zynsim/voxel.cpp
)
add_library(ZynNova::zynsim_voxel_core ALIAS zynnova_zynsim_voxel_core)
target_include_directories(zynnova_zynsim_voxel_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(zynnova_zynsim_voxel_core PUBLIC cxx_std_17)
set_target_properties(zynnova_zynsim_voxel_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
  target_compile_options(zynnova_zynsim_voxel_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(zynnova_zynsim_voxel_core PRIVATE -Wall -Wextra -Wpedantic)
endif()
if(OpenMP_CXX_FOUND)
  target_link_libraries(zynnova_zynsim_voxel_core PUBLIC OpenMP::OpenMP_CXX)
endif()
pybind11_add_module(_zynsim_voxel_native MODULE
  bindings/zynsim_voxel_module.cpp
)
target_link_libraries(_zynsim_voxel_native PRIVATE ZynNova::zynsim_voxel_core)
target_include_directories(_zynsim_voxel_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
install(TARGETS _zynsim_voxel_native
  LIBRARY DESTINATION zynnova/_native
  RUNTIME DESTINATION zynnova/_native
)

if(ZYNNOVA_BUILD_CPP_TESTS)
  add_executable(zynnova_zynsim_voxel_core_tests
    tests/test_zynsim_voxel.cpp
  )
  target_link_libraries(
    zynnova_zynsim_voxel_core_tests
    PRIVATE ZynNova::zynsim_voxel_core
  )
  add_test(
    NAME zynnova_zynsim_voxel_core_tests
    COMMAND zynnova_zynsim_voxel_core_tests
  )
endif()

# ZynSim phase-field C++/OpenMP kernels follow the shared project-level C++ layout.
add_library(zynnova_zynsim_phasefield_core STATIC
  src/zynsim/phasefield.cpp
)

add_library(ZynNova::zynsim_phasefield_core ALIAS zynnova_zynsim_phasefield_core)

target_include_directories(zynnova_zynsim_phasefield_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(zynnova_zynsim_phasefield_core PUBLIC cxx_std_17)
set_target_properties(
  zynnova_zynsim_phasefield_core
  PROPERTIES POSITION_INDEPENDENT_CODE ON
)

if(MSVC)
  target_compile_options(
    zynnova_zynsim_phasefield_core
    PRIVATE /W4 /permissive-
  )
else()
  target_compile_options(
    zynnova_zynsim_phasefield_core
    PRIVATE -Wall -Wextra -Wpedantic
  )
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(
    zynnova_zynsim_phasefield_core
    PUBLIC OpenMP::OpenMP_CXX
  )
endif()

pybind11_add_module(_zynsim_phasefield_native MODULE
  bindings/zynsim_phasefield_module.cpp
)

target_link_libraries(
  _zynsim_phasefield_native
  PRIVATE ZynNova::zynsim_phasefield_core
)
target_include_directories(
  _zynsim_phasefield_native
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
)

install(TARGETS _zynsim_phasefield_native
  LIBRARY DESTINATION zynnova/_native
  RUNTIME DESTINATION zynnova/_native
)

if(ZYNNOVA_BUILD_CPP_TESTS)
  add_executable(zynnova_zynsim_phasefield_core_tests
    tests/test_zynsim_phasefield.cpp
  )
  target_link_libraries(
    zynnova_zynsim_phasefield_core_tests
    PRIVATE ZynNova::zynsim_phasefield_core
  )
  add_test(
    NAME zynnova_zynsim_phasefield_core_tests
    COMMAND zynnova_zynsim_phasefield_core_tests
  )
endif()

endif()

# Experimental, dependency-isolated Ziver Kokkos numerical primitives.  This
# target is deliberately not a LAMMPS pair style or a Python extension: it is
# the native matrix-free building block that those integrations can share.
if(ZYNNOVA_BUILD_ZIVAR_KOKKOS)
  find_package(Kokkos CONFIG REQUIRED)

  add_library(zynnova_zivar_kokkos_core STATIC
    src/zivar/matrix_free.cpp
  )
  add_library(ZynNova::zivar_kokkos_core ALIAS zynnova_zivar_kokkos_core)

  target_include_directories(zynnova_zivar_kokkos_core
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )
  target_compile_features(zynnova_zivar_kokkos_core PUBLIC cxx_std_17)
  target_link_libraries(zynnova_zivar_kokkos_core PUBLIC Kokkos::kokkos)
  set_target_properties(
    zynnova_zivar_kokkos_core
    PROPERTIES POSITION_INDEPENDENT_CODE ON
  )

  if(MSVC)
    target_compile_options(
      zynnova_zivar_kokkos_core PRIVATE /W4 /permissive-
    )
  else()
    target_compile_options(
      zynnova_zivar_kokkos_core PRIVATE -Wall -Wextra -Wpedantic
    )
  endif()

  if(ZYNNOVA_BUILD_CPP_TESTS)
    add_executable(zynnova_zivar_kokkos_core_tests
      tests/test_zivar_kokkos.cpp
    )
    target_link_libraries(
      zynnova_zivar_kokkos_core_tests
      PRIVATE ZynNova::zivar_kokkos_core
    )
    add_test(
      NAME zynnova_zivar_kokkos_core_tests
      COMMAND zynnova_zivar_kokkos_core_tests
    )
  endif()
endif()
