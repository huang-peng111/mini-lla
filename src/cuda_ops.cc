#include "mini_llama/cuda_ops.h"

#ifndef MINI_LLA_USE_CUDA

#include <stdexcept>

namespace mini_lla {
namespace {

// CPU-only 构建下所有 GPU 算子统一报错。
[[noreturn]] void Disabled(const char* caller) {
  throw std::runtime_error(std::string(caller) +
                           ": CUDA 未启用,请使用 -DMINI_LLA_CUDA=ON 重新构建");
}

}  // namespace

CudaTensor CudaRmsNorm(const CudaTensor&, const CudaWeight&, float) {
  Disabled("CudaRmsNorm");
}

void CudaRope(CudaTensor*, CudaTensor*, int, float, int, int, int, RopeType) {
  Disabled("CudaRope");
}

CudaTensor CudaSilu(const CudaTensor&) { Disabled("CudaSilu"); }

CudaTensor CudaMul(const CudaTensor&, const CudaTensor&) {
  Disabled("CudaMul");
}

CudaTensor CudaAdd(const CudaTensor&, const CudaTensor&) {
  Disabled("CudaAdd");
}

CudaTensor CudaSoftmax(const CudaTensor&) { Disabled("CudaSoftmax"); }

CudaTensor CudaEmbeddingRow(const CudaWeight&, int) {
  Disabled("CudaEmbeddingRow");
}

CudaTensor CudaAddBias(const CudaTensor&, const CudaWeight*) {
  Disabled("CudaAddBias");
}

}  // namespace mini_lla

#endif  // MINI_LLA_USE_CUDA
