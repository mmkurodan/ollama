cmake_minimum_required(VERSION 3.22.1)

project("llama_jni")

set(GGML_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../llm/llama.cpp)

add_library(llama_jni SHARED
        jni/jni_llama.cpp)

target_include_directories(llama_jni PRIVATE
        ${GGML_ROOT_DIR}/include
        ${GGML_ROOT_DIR}/src
        ${GGML_ROOT_DIR}/src/ggml-cpu
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/base64/include)

target_link_libraries(llama_jni
        android
        log
        llama
        ggml)
