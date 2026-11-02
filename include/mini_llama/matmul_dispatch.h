#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_

#include "mini_llama/tensor.h"

namespace mini_lla {

// 矩阵乘的实现选择:朴素版本便于对拍,优化版本用于默认路径。
enum class MatmulMode { kNaive, kOptimized };

// 三重循环的朴素矩阵乘,作为正确性参考。
Tensor MatmulNaive(const Tensor& a, const Tensor& b);

// 调整循环顺序、让输入元素只读取一次的矩阵乘。
Tensor MatmulOptimized(const Tensor& a, const Tensor& b);

// 朴素线性层。
Tensor LinearNaive(const Tensor& x, const Tensor& weight);

// 优化线性层:逐行累加输出,适合 Transformer 的向量输入。
Tensor LinearOptimized(const Tensor& x, const Tensor& weight);

// 按指定模式执行矩阵乘或线性层。
Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode);
Tensor LinearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode);

// 读写全局默认模式,供命令行切换。
void SetMatmulMode(MatmulMode mode);
MatmulMode GetMatmulMode();

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_
