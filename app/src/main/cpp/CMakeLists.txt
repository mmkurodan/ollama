cmake_minimum_required(VERSION 3.22.1)
project(llama_android)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")

# llama.cpp のルート（gguf-v0.4.4）
set(LLAMA_ROOT_DIR ${CMAKE_SOURCE_DIR}/llama)

add_library(
    llama_jni
    SHARED

    # JNI
    jni/jni_llama.cpp

    # llama core
    ${LLAMA_ROOT_DIR}/llama.cpp

    # ggml core
    ${LLAMA_ROOT_DIR}/ggml.c
    ${LLAMA_ROOT_DIR}/ggml-alloc.c
    ${LLAMA_ROOT_DIR}/ggml-quants.c
    ${LLAMA_ROOT_DIR}/ggml-cpu.c

    # quant kernels
    ${LLAMA_ROOT_DIR}/k_quants.c
)

target_include_directories(
    llama_jni
    PRIVATE
    ${LLAMA_ROOT_DIR}

    ${CMAKE_SOURCE_DIR}/third_party/libcurl/include
    ${CMAKE_SOURCE_DIR}/third_party/mbedtls/include
)

# libcurl
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
