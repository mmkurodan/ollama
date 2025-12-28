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
    ${CMAKE_SOURCE_DIR}/third_party/libcurl/include
)

# libcurl (mbedTLS version)
add_library(curl STATIC IMPORTED)
set_target_properties(curl PROPERTIES IMPORTED_LOCATION
    ${CMAKE_SOURCE_DIR}/third_party/libcurl/lib/libcurl.a
)

target_link_libraries(
    llama_jni
    PRIVATE
    curl
    log
    android
    jnigraphics
    m
    atomic
)
