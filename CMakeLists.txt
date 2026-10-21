cmake_minimum_required(VERSION 3.5)

set(TARGET_NAME timestral)

# ONNX Runtime (CPU execution provider) from vcpkg, linked statically. See vcpkg_ports/README.md.
# DuckDB's tidy configure is not a vcpkg build, so the package is optional there.
if(CLANG_TIDY)
  find_package(onnxruntime CONFIG QUIET)
else()
  find_package(onnxruntime CONFIG REQUIRED)
endif()
# vcpkg installs the public headers under include/onnxruntime/; sources include them flat.
if(onnxruntime_FOUND)
  find_path(TIMESTRAL_ORT_INCLUDE_DIR NAMES onnxruntime_cxx_api.h PATH_SUFFIXES onnxruntime REQUIRED)
endif()

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})

set(CMAKE_CXX_STANDARD "17" CACHE STRING "C++ standard to enforce")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(src/include)

set(EXTENSION_SOURCES
    src/timestral_extension.cpp
    src/forecast_aggregate.cpp
    src/forecast_macro.cpp
    src/forecaster.cpp
    src/model_store.cpp
    src/time_grid.cpp)
# This file includes onnxruntime headers. Skip it when the tidy job has no ONNX Runtime.
if(onnxruntime_FOUND)
  list(APPEND EXTENSION_SOURCES src/ort_predictor.cpp)
endif()

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

if(onnxruntime_FOUND)
  foreach(_target ${EXTENSION_NAME} ${LOADABLE_EXTENSION_NAME})
    target_include_directories(${_target} PRIVATE ${TIMESTRAL_ORT_INCLUDE_DIR})
    target_link_libraries(${_target} onnxruntime::onnxruntime)
  endforeach()
endif()
# SHA-256 comes from DuckDB's bundled mbedtls (as in httpfs); the static extension gets it through duckdb_static.
target_link_libraries(${LOADABLE_EXTENSION_NAME} duckdb_mbedtls)

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
