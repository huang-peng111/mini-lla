#include "mini_llama/cuda_matmul.h"

#ifndef MINI_LLA_USE_CUDA

#include <stdexcept>

namespace mini_lla {

CudaTensor CudaLinear(const CudaTensor&, const CudaWeight&) {
  throw std::runtime_error(
      "CudaLinear: CUDA 未启用,请使用 -DMINI_LLA_CUDA=ON 重新构建");
}

}  // namespace mini_lla

#endif  // MINI_LLA_USE_CUDA
