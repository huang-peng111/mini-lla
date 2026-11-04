#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_CUDA_MATMUL_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_CUDA_MATMUL_H_

#include "mini_llama/cuda_tensor.h"
#include "mini_llama/cuda_weights.h"

namespace mini_lla {

// 设备上的线性层:y = x * W^T。
// x 的形状是 [weight.cols],返回长度为 weight.rows 的新张量。
// F32 权重走 cuBLAS,Q8_0/Q4_0/Q4_1 走自写的 block 级 GEMV。
CudaTensor CudaLinear(const CudaTensor& x, const CudaWeight& weight);

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_CUDA_MATMUL_H_
