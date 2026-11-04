cmake_minimum_required(VERSION 3.16)
project(mini_lla LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA 后端默认关闭;关闭时 CPU 构建不需要 CUDA Toolkit。
option(MINI_LLA_CUDA "Build the NVIDIA CUDA backend" OFF)

# 核心库:张量、算子、并发、量化、加载、前向和可选 CUDA 后端。
set(MINI_LLA_CORE_SOURCES
    src/tensor.cc
    src/ops.cc
    src/matmul_optimized.cc
    src/thread_pool.cc
    src/quantized_tensor.cc
    src/quant.cc
    src/model.cc
    src/kv_cache.cc
    src/batch.cc
    src/forward.cc
    src/cuda_runtime.cc
    src/cuda_tensor.cc
    src/cuda_weights.cc
    src/cuda_matmul.cc
    src/cuda_ops.cc
    src/cuda_attention.cc
    src/cuda_forward.cc
    src/tokenizer.cc
    src/gguf_tokenizer.cc
    src/sampler.cc
    src/benchmark.cc
    src/chat.cc
    src/prompt_builder.cc
    src/gguf.cc
    src/gguf_loader.cc
    src/loader.cc
    src/backend.cc)

add_library(mini_lla_core ${MINI_LLA_CORE_SOURCES})
target_include_directories(mini_lla_core
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MINI_LLA_CUDA)
  # 只有打开开关时才启用 CUDA 语言并查找 Toolkit。
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(mini_lla_core PRIVATE
      src/cuda_matmul.cu
      src/cuda_ops.cu
      src/cuda_attention.cu)
  target_compile_definitions(mini_lla_core PUBLIC MINI_LLA_USE_CUDA=1)
  target_link_libraries(mini_lla_core PUBLIC CUDA::cudart CUDA::cublas)
endif()

# 命令行程序。
add_executable(mini-lla src/main.cc)
target_link_libraries(mini-lla PRIVATE mini_lla_core)

# 测试程序:使用自带的极简 runner,不依赖外部框架。
add_executable(mini-lla-tests
    tests/test_main.cc
    tests/test_tensor.cc
    tests/test_ops.cc
    tests/test_thread_pool.cc
    tests/test_kv_cache.cc
    tests/test_forward.cc
    tests/test_loader.cc
    tests/test_tokenizer.cc
    tests/test_sampler_chat.cc
    tests/test_benchmark.cc
    tests/test_quant.cc
    tests/test_gguf.cc
    tests/test_gguf_loader.cc
    tests/test_cuda_runtime.cc)
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
