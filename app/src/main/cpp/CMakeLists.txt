cmake_minimum_required(VERSION 3.22.1)
project(llama_android)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -std=c++17")

add_subdirectory(llama)

add_library(
    llama_jni
    SHARED
    jni/jni_llama.cpp
)

target_include_directories(
    llama_jni
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/llama
)

target_link_libraries(
    llama_jni
    PRIVATE
    llama
    log
    android
    jnigraphics
    m
    atomic
)
