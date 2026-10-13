#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_

#include "mini_llama/tensor.h"
namespace mini_lla {

// 矩阵乘实现模式：朴素版本便于调试，优化版本用于默认路径。
enum class MatmulMode { kNaive, kOptimized };
Tensor matmul_naive(const Tensor&, const Tensor&);
Tensor matmul_optimized(const Tensor&, const Tensor&);
Tensor matmul_dispatch(const Tensor&, const Tensor&,
                       MatmulMode = MatmulMode::kOptimized);
void set_matmul_mode(MatmulMode);
MatmulMode matmul_mode();
}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_
