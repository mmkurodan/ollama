cmake_minimum_required(VERSION 3.22.1)
project(llama_android)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")

# llama.cpp のルート（gguf-0.17.1）
set(LLAMA_ROOT_DIR ${CMAKE_SOURCE_DIR}/llama)
set(GGML_ROOT_DIR ${CMAKE_SOURCE_DIR}/ggml)

# Collect all llama source files from src directory
file(GLOB LLAMA_SOURCES
    ${LLAMA_ROOT_DIR}/src/*.cpp
)

# Collect ggml source files
set(GGML_SOURCES
    ${GGML_ROOT_DIR}/src/ggml.c
    ${GGML_ROOT_DIR}/src/ggml-alloc.c
    ${GGML_ROOT_DIR}/src/ggml-backend.cpp
    ${GGML_ROOT_DIR}/src/ggml-quants.c
    ${GGML_ROOT_DIR}/src/ggml-cpu/ggml-cpu.cpp
    ${GGML_ROOT_DIR}/src/ggml-cpu/ggml-cpu-traits.cpp
    ${GGML_ROOT_DIR}/src/ggml-cpu/ggml-cpu-aarch64.cpp
    ${GGML_ROOT_DIR}/src/ggml-cpu/quants.c
)

add_library(
    llama_jni
    SHARED

    # JNI
    jni/jni_llama.cpp

    # llama core sources
    ${LLAMA_SOURCES}
    
    # ggml core sources
    ${GGML_SOURCES}
)

target_include_directories(
    llama_jni
    PRIVATE
    ${LLAMA_ROOT_DIR}/include
    ${LLAMA_ROOT_DIR}/src
    ${GGML_ROOT_DIR}/include
    ${GGML_ROOT_DIR}/src

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
        ${CMAKE_SOURCE_DIR}/third_party/mbedcrypto/lib/libmbedcrypto.a
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
