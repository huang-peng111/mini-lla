#include "mini_llama/matmul_dispatch.h"
#include <stdexcept>
namespace mini_lla {
static MatmulMode g_mode = MatmulMode::kOptimized;

static void CheckShapes(const Tensor& a, const Tensor& b) {
  if (a.shape.size() != 2 || b.shape.size() != 2 || a.shape[1] != b.shape[0]) {
    throw std::invalid_argument("matmul: expected [m,k] x [k,n]");
  }
}

Tensor matmul_naive(const Tensor& a, const Tensor& b) {
  CheckShapes(a, b);
  Tensor output({a.shape[0], b.shape[1]});
  for (int row = 0; row < a.shape[0]; ++row) {
    for (int column = 0; column < b.shape[1]; ++column) {
      for (int inner = 0; inner < a.shape[1]; ++inner) {
        output.at2(row, column) += a.at2(row, inner) * b.at2(inner, column);
      }
    }
  }
  return output;
}

Tensor matmul_optimized(const Tensor& a, const Tensor& b) {
  CheckShapes(a, b);
  Tensor output({a.shape[0], b.shape[1]});
  // 调整循环顺序，让输入 A 的元素只读取一次，减少重复访存。
  for (int row = 0; row < a.shape[0]; ++row) {
    for (int inner = 0; inner < a.shape[1]; ++inner) {
      const float value = a.at2(row, inner);
      for (int column = 0; column < b.shape[1]; ++column) {
        output.at2(row, column) += value * b.at2(inner, column);
      }
    }
  }
  return output;
}

Tensor matmul_dispatch(const Tensor& a, const Tensor& b, MatmulMode mode) {
  return mode == MatmulMode::kNaive ? matmul_naive(a, b)
                                    : matmul_optimized(a, b);
}

void set_matmul_mode(MatmulMode mode) { g_mode = mode; }

MatmulMode matmul_mode() { return g_mode; }
}
