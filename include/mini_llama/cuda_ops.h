#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_CUDA_OPS_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_CUDA_OPS_H_

#include "mini_llama/cuda_tensor.h"
#include "mini_llama/cuda_weights.h"
#include "mini_llama/ops.h"

namespace mini_lla {

// 对设备向量做 RMSNorm,weight 是设备上的一维权重。
CudaTensor CudaRmsNorm(const CudaTensor& x, const CudaWeight& weight,
                       float epsilon);

// 在设备上对 q/k 施加 RoPE,q 与 k 的形状是 [heads, head_dim]。
void CudaRope(CudaTensor* q, CudaTensor* k, int position, float theta,
              int n_heads, int n_kv_heads, int head_dim, RopeType rope_type);

// 逐元素 SiLU。
CudaTensor CudaSilu(const CudaTensor& x);

// 两个同形状张量逐元素相乘。
CudaTensor CudaMul(const CudaTensor& a, const CudaTensor& b);

// 两个同形状张量逐元素相加。
CudaTensor CudaAdd(const CudaTensor& a, const CudaTensor& b);

// 对一维向量做 softmax。
CudaTensor CudaSoftmax(const CudaTensor& x);

// 从量化 embedding 中取出第 token 行,返回长度 cols 的向量。
CudaTensor CudaEmbeddingRow(const CudaWeight& embedding, int token);

// 给向量加上可选偏置,bias 为空时只做拷贝。
CudaTensor CudaAddBias(const CudaTensor& value, const CudaWeight* bias);

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_CUDA_OPS_H_
