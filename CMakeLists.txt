cmake_minimum_required(VERSION 3.16)
project(mini_lla LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 核心库:张量、算子、量化、模型加载和 Transformer 前向。
add_library(mini_lla_core
    src/tensor.cc
    src/ops.cc
    src/matmul_optimized.cc
    src/quantized_tensor.cc
    src/quant.cc
    src/model.cc
    src/kv_cache.cc
    src/batch.cc
    src/forward.cc
    src/tokenizer.cc
    src/gguf_tokenizer.cc
    src/sampler.cc
    src/chat.cc
    src/prompt_builder.cc
    src/gguf.cc
    src/gguf_loader.cc
    src/loader.cc)
target_include_directories(mini_lla_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 命令行程序。
add_executable(mini-lla src/main.cc)
target_link_libraries(mini-lla PRIVATE mini_lla_core)

# 测试程序:使用自带的极简 runner,不依赖外部框架。
add_executable(mini-lla-tests
    tests/test_main.cc
    tests/test_tensor.cc
    tests/test_ops.cc
    tests/test_kv_cache.cc
    tests/test_forward.cc
    tests/test_loader.cc
    tests/test_tokenizer.cc
    tests/test_sampler_chat.cc
    tests/test_quant.cc
    tests/test_gguf.cc
    tests/test_gguf_loader.cc)
target_link_libraries(mini-lla-tests PRIVATE mini_lla_core)

if(MSVC)
  # /utf-8 是必需的:源码中的中文注释必须按 UTF-8 解析。
  foreach(target mini_lla_core mini-lla mini-lla-tests)
    target_compile_options(${target} PRIVATE /W4 /utf-8)
  endforeach()
else()
  foreach(target mini_lla_core mini-lla mini-lla-tests)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  endforeach()
endif()

enable_testing()
add_test(NAME mini-lla-tests COMMAND mini-lla-tests
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(
    NAME cli-smoke
    COMMAND ${CMAKE_COMMAND} -E env
            MINI_LLA_BIN=$<TARGET_FILE:mini-lla>
            ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/test_cli.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()
