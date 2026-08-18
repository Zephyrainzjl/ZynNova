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


# ---------------------------------------------------------------------------
# ZynMorph adaptive TetGen 1.6.0 kernel.
#
# TetGen itself is AGPL-3.0-or-later.  The ZynNova pybind11 adapter remains
# MIT, but a wheel linked with and distributing TetGen is subject to TetGen's
# license.  Source/license files are installed beside the extension.
# ---------------------------------------------------------------------------
if(ZYNNOVA_BUILD_TETGEN)
  set(_zynnova_tetgen_vendored "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tetgen/source")
  if(ZYNNOVA_TETGEN_SOURCE_DIR)
    set(_zynnova_tetgen_source "${ZYNNOVA_TETGEN_SOURCE_DIR}")
  elseif(EXISTS "${_zynnova_tetgen_vendored}/tetgen.cxx")
    set(_zynnova_tetgen_source "${_zynnova_tetgen_vendored}")
  elseif(ZYNNOVA_TETGEN_ALLOW_FETCH)
    include(FetchContent)
    message(STATUS
      "TetGen source is not vendored; fetching pinned pyvista/tetgen commit "
      "c039698cf4cce5c671b281c003dbc6cd8e58acc3 (TetGen 1.6.0, AGPL)."
    )
    FetchContent_Declare(
      zynnova_tetgen_upstream
      GIT_REPOSITORY https://github.com/pyvista/tetgen.git
      GIT_TAG c039698cf4cce5c671b281c003dbc6cd8e58acc3
      GIT_SHALLOW FALSE
      GIT_PROGRESS TRUE
      SOURCE_SUBDIR zynnova-no-upstream-cmake
    )
    FetchContent_MakeAvailable(zynnova_tetgen_upstream)
    set(_zynnova_tetgen_source "${zynnova_tetgen_upstream_SOURCE_DIR}/src")
  else()
    message(FATAL_ERROR
      "ZYNNOVA_BUILD_TETGEN=ON but TetGen source is absent. Run "
      "`python scripts/vendor_tetgen.py --accept-agpl`, set "
      "ZYNNOVA_TETGEN_SOURCE_DIR, enable ZYNNOVA_TETGEN_ALLOW_FETCH, or "
      "configure -DZYNNOVA_BUILD_TETGEN=OFF."
    )
  endif()

  foreach(_required tetgen.cxx tetgen.h predicates.cxx tetgen-license)
    if(NOT EXISTS "${_zynnova_tetgen_source}/${_required}")
      message(FATAL_ERROR
        "Incomplete TetGen source at ${_zynnova_tetgen_source}: missing ${_required}"
      )
    endif()
  endforeach()

  add_library(zynnova_tetgen_core STATIC
    "${_zynnova_tetgen_source}/tetgen.cxx"
    "${_zynnova_tetgen_source}/predicates.cxx"
  )
  add_library(ZynNova::tetgen_core ALIAS zynnova_tetgen_core)
  target_include_directories(zynnova_tetgen_core PUBLIC "${_zynnova_tetgen_source}")
  target_compile_definitions(zynnova_tetgen_core PUBLIC TETLIBRARY)
  target_compile_features(zynnova_tetgen_core PUBLIC cxx_std_17)
  set_target_properties(
    zynnova_tetgen_core
    PROPERTIES POSITION_INDEPENDENT_CODE ON
  )
  if(MSVC)
    target_compile_options(zynnova_tetgen_core PRIVATE /O2 /W0)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(
      zynnova_tetgen_core PRIVATE -O3 -fno-expensive-optimizations -w
    )
  else()
    target_compile_options(zynnova_tetgen_core PRIVATE -O3 -w)
  endif()

  pybind11_add_module(_zynmorph_tetgen_native MODULE
    bindings/zynmorph_tetgen_module.cpp
  )
  target_link_libraries(
    _zynmorph_tetgen_native PRIVATE ZynNova::tetgen_core
  )
  # Treat upstream TetGen headers as system headers so project warning levels
  # do not surface harmless warnings from third-party code (notably MinGW).
  target_include_directories(
    _zynmorph_tetgen_native SYSTEM PRIVATE "${_zynnova_tetgen_source}"
  )
  target_compile_features(_zynmorph_tetgen_native PRIVATE cxx_std_17)
  if(MSVC)
    target_compile_options(_zynmorph_tetgen_native PRIVATE /W4 /permissive-)
  else()
    target_compile_options(
      _zynmorph_tetgen_native PRIVATE -Wall -Wextra -Wpedantic
    )
  endif()

  # MinGW-built .pyd files can compile/link successfully yet fail to import in
  # a Jupyter/Python process that does not inherit the compiler's PATH.  Link
  # the GCC/C++ runtimes statically when possible and also co-install any
  # compiler runtime DLLs that the toolchain reports, next to the extension.
  set(_zynnova_tetgen_mingw_runtime_files)
  if(MINGW)
    target_link_options(
      _zynmorph_tetgen_native PRIVATE -static-libgcc -static-libstdc++
    )
    foreach(_runtime_name IN ITEMS
        libwinpthread-1.dll
        libgcc_s_seh-1.dll
        libgcc_s_sjlj-1.dll
        libgcc_s_dw2-1.dll
        libstdc++-6.dll
    )
      execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" "-print-file-name=${_runtime_name}"
        OUTPUT_VARIABLE _runtime_path
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
      if(IS_ABSOLUTE "${_runtime_path}" AND EXISTS "${_runtime_path}")
        list(APPEND _zynnova_tetgen_mingw_runtime_files "${_runtime_path}")
      endif()
    endforeach()
    list(REMOVE_DUPLICATES _zynnova_tetgen_mingw_runtime_files)
  endif()

  install(TARGETS _zynmorph_tetgen_native
    LIBRARY DESTINATION zynnova/_native
    RUNTIME DESTINATION zynnova/_native
  )
  if(_zynnova_tetgen_mingw_runtime_files)
    install(
      FILES ${_zynnova_tetgen_mingw_runtime_files}
      DESTINATION zynnova/_native
    )
  endif()
  # Package the exact TetGen source used to build the extension.  Only files
  # required for license/source traceability are mandatory.  Optional project
  # documentation and the vendor manifest are installed only when present;
  # this keeps editable/wheel installs robust after vendor_tetgen.py replaces
  # the source staging directory and on Windows/CMake installations.
  set(_zynnova_tetgen_distribution_files
    "${_zynnova_tetgen_source}/tetgen.cxx"
    "${_zynnova_tetgen_source}/tetgen.h"
    "${_zynnova_tetgen_source}/predicates.cxx"
    "${_zynnova_tetgen_source}/tetgen-license"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tetgen/SOURCE_LOCK.json"
  )

  set(_zynnova_tetgen_optional_distribution_files
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tetgen/README.md"
    "${_zynnova_tetgen_source}/VENDOR_MANIFEST.json"
  )
  foreach(_optional_file IN LISTS _zynnova_tetgen_optional_distribution_files)
    if(EXISTS "${_optional_file}" AND NOT IS_DIRECTORY "${_optional_file}")
      list(APPEND _zynnova_tetgen_distribution_files "${_optional_file}")
    endif()
  endforeach()

  install(
    FILES ${_zynnova_tetgen_distribution_files}
    DESTINATION zynnova/_third_party/tetgen
  )
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
