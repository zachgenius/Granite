cmake_minimum_required(VERSION 3.22.1)
project(GraniteDemo LANGUAGES C CXX)

set(GRANITE_ROOT "" CACHE PATH "Path to Granite repository root")
if(NOT GRANITE_ROOT)
    message(FATAL_ERROR "GRANITE_ROOT must be set via -DGRANITE_ROOT=...")
endif()

set(GRANITE_BUILD_VULKAN ON CACHE BOOL "" FORCE)
set(GRANITE_BUILD_METAL OFF CACHE BOOL "" FORCE)
set(GRANITE_BUILD_CPU ON CACHE BOOL "" FORCE)
set(GRANITE_BUILD_COREML OFF CACHE BOOL "" FORCE)
set(GRANITE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GRANITE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory("${GRANITE_ROOT}" "${CMAKE_BINARY_DIR}/granite")

add_library(granite_jni SHARED
    "${GRANITE_ROOT}/src/android/granite_jni.cpp"
)

target_include_directories(granite_jni PRIVATE
    "${GRANITE_ROOT}/include"
)

target_link_libraries(granite_jni
    PRIVATE
        granite
        android
        log
)
