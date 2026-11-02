#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_OPS_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_OPS_H_

#include "mini_llama/tensor.h"

namespace mini_lla {

// RoPE 的两种配对方式:LLaMA 用相邻通道配对,NeoX/Qwen2 用前后半配对。
enum class RopeType { kNormal, kNeoX };

// 矩阵乘:[m, k] x [k, n] -> [m, n]。
Tensor Matmul(const Tensor& a, const Tensor& b);

// 线性层 y = x * weight^T,weight 形状为 [输出维度, 输入维度]。
// 输入可以是一维向量或 [行数, 输入维度],输出保持相同的秩。
Tensor Linear(const Tensor& x, const Tensor& weight);

// RMSNorm:y = x / sqrt(mean(x^2) + eps) * weight。
Tensor RmsNorm(const Tensor& x, const Tensor& weight, float epsilon);

// Softmax:先减去最大值再求指数,避免溢出一维向量。
Tensor Softmax(const Tensor& x);

// SiLU 激活:x * sigmoid(x)。
Tensor Silu(const Tensor& x);

// SwiGLU:Silu(gate) 与 up 逐元素相乘。
Tensor SwiGlu(const Tensor& gate, const Tensor& up);

// 两个同形状张量逐元素相加(残差连接)。
Tensor Add(const Tensor& a, const Tensor& b);

// 对 q/k 按 head 施加旋转位置编码,q 和 k 的形状为 [heads, head_dim]。
void Rope(Tensor& q, Tensor& k, int position, float theta, RopeType rope_type);

// 返回一维张量中最大值所在的下标。
int ArgMax(const Tensor& x);

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_OPS_H_
