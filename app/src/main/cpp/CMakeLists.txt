cmake_minimum_required(VERSION 3.22.1)
project(llama_android)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -std=c++17")

add_library(
    llama_jni
    SHARED
    jni/jni_llama.cpp
)

# third_party include
include_directories(
    ${CMAKE_SOURCE_DIR}/app/src/main/cpp/third_party/libcurl/include
    ${CMAKE_SOURCE_DIR}/app/src/main/cpp/third_party/openssl/include
)

# libcurl
add_library(curl STATIC IMPORTED)
set_target_properties(curl PROPERTIES IMPORTED_LOCATION
    ${CMAKE_SOURCE_DIR}/app/src/main/cpp/third_party/libcurl/lib/libcurl.a
)

# OpenSSL
add_library(ssl STATIC IMPORTED)
set_target_properties(ssl PROPERTIES IMPORTED_LOCATION
    ${CMAKE_SOURCE_DIR}/app/src/main/cpp/third_party/openssl/lib/libssl.a
)

add_library(crypto STATIC IMPORTED)
set_target_properties(crypto PROPERTIES IMPORTED_LOCATION
    ${CMAKE_SOURCE_DIR}/app/src/main/cpp/third_party/openssl/lib/libcrypto.a
)

target_link_libraries(
    llama_jni
    PRIVATE
    curl
    ssl
    crypto
    log
    android
    jnigraphics
    m
    atomic
)
