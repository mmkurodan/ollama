cmake_minimum_required(VERSION 3.22.1)
project(llama_android)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -std=c++17")

# llama.cpp のパス（光男のプロジェクト構成に合わせて調整）
set(LLAMA_DIR ${CMAKE_SOURCE_DIR}/llama)

# llama.cpp / ggml.c / ggml-alloc.c / ggml-backend.c を追加
add_library(
    llama_jni
    SHARED
    jni/jni_llama.cpp
    ${LLAMA_DIR}/llama.cpp
    ${LLAMA_DIR}/ggml.c
    ${LLAMA_DIR}/ggml-alloc.c
    ${LLAMA_DIR}/ggml-backend.c
)

# include path に llama ディレクトリを追加
target_include_directories(
    llama_jni
    PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/libcurl/include
    ${CMAKE_SOURCE_DIR}/third_party/mbedtls/include
    ${LLAMA_DIR}
)

# libcurl (mbedTLS version)
add_library(curl STATIC IMPORTED)
set_target_properties(curl PROPERTIES IMPORTED_LOCATION
    ${CMAKE_SOURCE_DIR}/third_party/libcurl/lib/libcurl.a
)

# mbedTLS
add_library(mbedtls STATIC IMPORTED)
set_target_properties(mbedtls PROPERTIES IMPORTED_LOCATION
    ${CMAKE_SOURCE_DIR}/third_party/mbedtls/lib/libmbedtls.a
)

add_library(mbedcrypto STATIC IMPORTED)
set_target_properties(mbedcrypto PROPERTIES IMPORTED_LOCATION
    ${CMAKE_SOURCE_DIR}/third_party/mbedtls/lib/libmbedcrypto.a
)

add_library(mbedx509 STATIC IMPORTED)
set_target_properties(mbedx509 PROPERTIES IMPORTED_LOCATION
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
