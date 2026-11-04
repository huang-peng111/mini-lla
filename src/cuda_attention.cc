#include "mini_llama/cuda_attention.h"

#ifndef MINI_LLA_USE_CUDA

#include <stdexcept>
#include <string>

namespace mini_lla {
namespace {

// CPU-only 构建下所有 GPU 注意力接口统一报错。
[[noreturn]] void Disabled(const char* caller) {
  throw std::runtime_error(std::string(caller) +
                           ": CUDA 未启用,请使用 -DMINI_LLA_CUDA=ON 重新构建");
}

}  // namespace

void CudaKvCache::Reset(int, int, int, int) { Disabled("CudaKvCache::Reset"); }

void CudaKvCache::Write(int, int, const CudaTensor&, const CudaTensor&) {
  Disabled("CudaKvCache::Write");
}

void CudaKvCache::Clear() {}

const float* CudaKvCache::Keys(int) const { Disabled("CudaKvCache::Keys"); }

const float* CudaKvCache::Values(int) const {
  Disabled("CudaKvCache::Values");
}

CudaTensor CudaAttention(const CudaTensor&, const CudaKvCache&, int, int, int,
                         int, int) {
  Disabled("CudaAttention");
}

}  // namespace mini_lla

#endif  // MINI_LLA_USE_CUDA
