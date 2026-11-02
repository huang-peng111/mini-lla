#include "mini_llama/matmul_dispatch.h"

#include <stdexcept>

namespace mini_lla {
namespace {

// 全局默认模式,默认为优化实现。
MatmulMode g_mode = MatmulMode::kOptimized;

// 检查矩阵乘的两个输入形状是否匹配。
void CheckMatmulShapes(const Tensor& a, const Tensor& b) {
  if (a.Rank() != 2 || b.Rank() != 2 || a.shape[1] != b.shape[0]) {
    throw std::invalid_argument("matmul: expected [m, k] x [k, n]");
  }
}

// 检查线性层权重和输入形状,并返回输入行数。
int CheckLinearShapes(const Tensor& x, const Tensor& weight) {
  if (weight.Rank() != 2) {
    throw std::invalid_argument("linear: weight must be 2D");
  }
  if (x.Rank() == 1) {
    if (x.shape[0] != weight.shape[1]) {
      throw std::invalid_argument("linear: input size mismatch");
    }
    return 1;
  }
  if (x.Rank() == 2 && x.shape[1] == weight.shape[1]) {
    return x.shape[0];
  }
  throw std::invalid_argument("linear: expected x as [k] or [m, k]");
}

}  // namespace

Tensor MatmulNaive(const Tensor& a, const Tensor& b) {
  CheckMatmulShapes(a, b);
  Tensor output({a.shape[0], b.shape[1]});
  for (int row = 0; row < a.shape[0]; ++row) {
    for (int column = 0; column < b.shape[1]; ++column) {
      float sum = 0.0f;
      for (int inner = 0; inner < a.shape[1]; ++inner) {
        sum += a.At2(row, inner) * b.At2(inner, column);
      }
      output.At2(row, column) = sum;
    }
  }
  return output;
}

Tensor MatmulOptimized(const Tensor& a, const Tensor& b) {
  CheckMatmulShapes(a, b);
  Tensor output({a.shape[0], b.shape[1]});
  for (int row = 0; row < a.shape[0]; ++row) {
    for (int inner = 0; inner < a.shape[1]; ++inner) {
      const float value = a.At2(row, inner);
      for (int column = 0; column < b.shape[1]; ++column) {
        output.At2(row, column) += value * b.At2(inner, column);
      }
    }
  }
  return output;
}

Tensor LinearNaive(const Tensor& x, const Tensor& weight) {
  const int rows = CheckLinearShapes(x, weight);
  const bool is_vector = x.Rank() == 1;
  Tensor output(is_vector ? std::vector<int>{weight.shape[0]}
                          : std::vector<int>{rows, weight.shape[0]});
  for (int row = 0; row < rows; ++row) {
    for (int out = 0; out < weight.shape[0]; ++out) {
      float sum = 0.0f;
      for (int k = 0; k < weight.shape[1]; ++k) {
        sum += x.data[static_cast<size_t>(row) * weight.shape[1] + k] *
               weight.At2(out, k);
      }
      output.data[static_cast<size_t>(row) * weight.shape[0] + out] = sum;
    }
  }
  return output;
}

Tensor LinearOptimized(const Tensor& x, const Tensor& weight) {
  const int rows = CheckLinearShapes(x, weight);
  const bool is_vector = x.Rank() == 1;
  const int in_features = weight.shape[1];
  Tensor output(is_vector ? std::vector<int>{weight.shape[0]}
                          : std::vector<int>{rows, weight.shape[0]});
  for (int row = 0; row < rows; ++row) {
    const float* source = x.data.data() + static_cast<size_t>(row) * in_features;
    float* destination =
        output.data.data() + static_cast<size_t>(row) * weight.shape[0];
    for (int out = 0; out < weight.shape[0]; ++out) {
      const float* filters = weight.RowPtr(out);
      float sum = 0.0f;
      for (int k = 0; k < in_features; ++k) {
        sum += source[k] * filters[k];
      }
      destination[out] = sum;
    }
  }
  return output;
}

Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode) {
  return mode == MatmulMode::kNaive ? MatmulNaive(a, b) : MatmulOptimized(a, b);
}

Tensor LinearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode) {
  return mode == MatmulMode::kNaive ? LinearNaive(x, weight)
                                    : LinearOptimized(x, weight);
}

void SetMatmulMode(MatmulMode mode) { g_mode = mode; }

MatmulMode GetMatmulMode() { return g_mode; }

}  // namespace mini_lla
