cmake_minimum_required(VERSION 3.22.1)
project(llama_android)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 最適化・ポータビリティ
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -std=c++17")

# llama.cpp のルート
set(LLAMA_ROOT_DIR        ${CMAKE_SOURCE_DIR}/llama)
set(LLAMA_INCLUDE_DIR     ${LLAMA_ROOT_DIR}/include)
set(LLAMA_SRC_DIR         ${LLAMA_ROOT_DIR}/src)

# ggml のルート
set(GGML_ROOT_DIR         ${LLAMA_ROOT_DIR}/ggml)
set(GGML_INCLUDE_DIR      ${GGML_ROOT_DIR}/include)
set(GGML_SRC_DIR          ${GGML_ROOT_DIR}/src)

add_library(
    llama_jni
    SHARED

    # JNI
    jni/jni_llama.cpp

    # llama.cpp 本体
    ${LLAMA_SRC_DIR}/llama.cpp
    ${LLAMA_SRC_DIR}/models/llama.cpp

    # ggml core
    ${GGML_SRC_DIR}/ggml.c
    ${GGML_SRC_DIR}/ggml-alloc.c
    ${GGML_SRC_DIR}/ggml-quants.c

    # ggml CPU backend
    ${GGML_SRC_DIR}/ggml-cpu/ggml-cpu.c
)

target_include_directories(
    llama_jni
    PRIVATE

    # llama.cpp
    ${LLAMA_INCLUDE_DIR}
    ${LLAMA_SRC_DIR}
    ${LLAMA_SRC_DIR}/models

    # ggml
    ${GGML_INCLUDE_DIR}
    ${GGML_SRC_DIR}

    # third_party
    ${CMAKE_SOURCE_DIR}/third_party/libcurl/include
    ${CMAKE_SOURCE_DIR}/third_party/mbedtls/include
)

# libcurl (mbedTLS 版を想定)
add_library(curl STATIC IMPORTED)
set_target_properties(curl PROPERTIES
    IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/third_party/libcurl/lib/libcurl.a
)

# mbedTLS
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
