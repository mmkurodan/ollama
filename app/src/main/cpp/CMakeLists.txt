cmake_minimum_required(VERSION 3.22.1)
project(llama_android)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")

# LLAMA_ROOT_DIR (points to app/src/main/cpp/llama)
set(LLAMA_ROOT_DIR ${CMAKE_SOURCE_DIR}/llama)
set(GGML_ROOT_DIR ${CMAKE_SOURCE_DIR}/llama/ggml)

# ------------------------------------------------------------------
# Collect only the implementation files that actually exist in the
# repository layout under app/src/main/cpp/llama.
# We intentionally ignore any previous (incorrect) file lists.
# If additional implementation files are required by build errors,
# add them explicitly here.
# ------------------------------------------------------------------

# Root-level llama implementation files (e.g. llama.cpp)
file(GLOB LLAMA_ROOT_FILES
    ${LLAMA_ROOT_DIR}/*.cpp
)

# common implementation files used by core
file(GLOB LLAMA_COMMON_FILES
    ${LLAMA_ROOT_DIR}/common/*.cpp
)

# src implementations (if present)
file(GLOB LLAMA_SRC_FILES
    ${LLAMA_ROOT_DIR}/src/*.cpp
)

set(LLAMA_SOURCES
    ${LLAMA_ROOT_FILES}
    ${LLAMA_COMMON_FILES}
    ${LLAMA_SRC_FILES}
)

# ------------------------------------------------------------------
# Collect ggml sources that exist under llama/ggml/src
# Allow both .c and .cpp files and cpu/ subdir implementations.
# ------------------------------------------------------------------
file(GLOB GGML_C_SOURCES
    ${GGML_ROOT_DIR}/src/*.c
    ${GGML_ROOT_DIR}/src/ggml-*.c
)

file(GLOB GGML_CPP_SOURCES
    ${GGML_ROOT_DIR}/src/*.cpp
    ${GGML_ROOT_DIR}/src/ggml-*.cpp
    ${GGML_ROOT_DIR}/src/ggml-cpu/*.cpp
)

set(GGML_SOURCES
    ${GGML_C_SOURCES}
    ${GGML_CPP_SOURCES}
)

# JNI shared library
add_library(
    llama_jni
    SHARED

    # JNI
    jni/jni_llama.cpp

    # llama core sources (only those found above)
    ${LLAMA_SOURCES}

    # ggml core sources (only those found above)
    ${GGML_SOURCES}
)

# Include directories required by JNI and llama.h
target_include_directories(
    llama_jni
    PRIVATE
    ${LLAMA_ROOT_DIR}/include    # <- important: solves #include "llama.h"
    ${LLAMA_ROOT_DIR}/src
    ${LLAMA_ROOT_DIR}/common
    ${GGML_ROOT_DIR}/include
    ${GGML_ROOT_DIR}/src

    ${CMAKE_SOURCE_DIR}/third_party/libcurl/include
    ${CMAKE_SOURCE_DIR}/third_party/mbedtls/include
)

# Link prebuilt third-party libraries
add_library(curl STATIC IMPORTED)
set_target_properties(curl PROPERTIES
    IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/third_party/libcurl/lib/libcurl.a
)

add_library(mbedtls STATIC IMPORTED)
set_target_properties(mbedtls PROPERTIES
    IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/third_party/mbedtls/lib/libmbedtls.a
)

add_library(mbedcrypto STATIC IMPORTED)
set_target_properties(mbedcrypto PROPERTIES
    IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/third_party/mbedtls/lib/libmbedcrypto.a
)

add_library(mbedx509 STATIC IMPORTED)
set_target_properties(mbedx509 PROPERTIES
    IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/third_party/mbedtls/lib/libmbedx509.a
)

target_link_libraries(
    llama_jni
    PRIVATE
    curl
    mbedtls
    mbedcrypto
    mbedx509

    log
    android
    jnigraphics
    m
    atomic
)
