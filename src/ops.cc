#include "mini_llama/ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "mini_llama/matmul_dispatch.h"
namespace mini_lla {
// 检查逐元素算子的两个输入形状是否一致。
static void RequireSameShape(const Tensor& a, const Tensor& b,
                             const char* name) {
  if (a.shape != b.shape) {
    throw std::invalid_argument(std::string(name) + ": shape mismatch");
  }
}

// 使用当前调度模式执行矩阵乘。
Tensor matmul(const Tensor& a, const Tensor& b) {
  return matmul_dispatch(a, b, matmul_mode());
}

// 执行 Transformer 中使用的二维线性层。
Tensor linear(const Tensor& x, const Tensor& weight, const Tensor* bias) {
  if (x.shape.size() != 2 || weight.shape.size() != 2 ||
      x.shape[1] != weight.shape[1]) {
    throw std::invalid_argument("linear: expected x [m,k], weight [n,k]");
  }
  Tensor output({x.shape[0], weight.shape[0]});
  for (int row = 0; row < x.shape[0]; ++row) {
    for (int out = 0; out < weight.shape[0]; ++out) {
      for (int k = 0; k < x.shape[1]; ++k) {
        output.at2(row, out) += x.at2(row, k) * weight.at2(out, k);
      }
      if (bias != nullptr) output.at2(row, out) += bias->at1(out);
    }
  }
  return output;
}

Tensor rms_norm(const Tensor& x, const Tensor& weight, float epsilon) {
  // RMSNorm 用均方根缩放向量，不减去均值，适合 Transformer 隐状态。
  // RMSNorm 使用均方根缩放向量，不减去均值。
  RequireSameShape(x, weight, "rms_norm");
  float square_sum = 0.0f;
  for (float value : x.data) square_sum += value * value;
  float inverse_rms = 1.0f / std::sqrt(square_sum / x.size() + epsilon);
  Tensor output(x.shape);
  for (size_t i = 0; i < x.size(); ++i) {
    output.data[i] = x.data[i] * inverse_rms * weight.data[i];
  }
  return output;
}

Tensor softmax(const Tensor& x) {
  // 先减去最大值，避免 exp 在较大 logits 上溢出。
  // 先减去最大值，避免 exp 在较大 logits 上溢出。
  if (x.shape.size() != 1 || x.data.empty()) {
    throw std::invalid_argument("softmax: expected non-empty vector");
  }
  float maximum = *std::max_element(x.data.begin(), x.data.end());
  Tensor output(x.shape);
  float sum = 0.0f;
  for (size_t i = 0; i < x.size(); ++i) {
    output.data[i] = std::exp(x.data[i] - maximum);
    sum += output.data[i];
  }
  for (float& value : output.data) value /= sum;
  return output;
}

// 对张量逐元素计算 SiLU。
Tensor silu(const Tensor& x) {
  Tensor output(x.shape);
  for (size_t i = 0; i < x.size(); ++i) {
    output.data[i] = x.data[i] / (1.0f + std::exp(-x.data[i]));
  }
  return output;
}

// 将 gate 的 SiLU 结果与 up 做逐元素相乘。
Tensor swiglu(const Tensor& gate, const Tensor& up) {
  RequireSameShape(gate, up, "swiglu");
  Tensor output = silu(gate);
  for (size_t i = 0; i < output.size(); ++i) output.data[i] *= up.data[i];
  return output;
}

void rope(Tensor& q, Tensor& k, int position, float theta) {
  // RoPE 将每两个通道视为二维平面，按 token 位置旋转 Q/K。
  // RoPE 将每两个通道视为二维平面，按 token 位置旋转 Q/K。
  if (q.shape.size() != 2 || k.shape.size() != 2 || q.shape[1] != k.shape[1] ||
      q.shape[1] % 2 != 0) {
    throw std::invalid_argument("rope: expected [heads, even_dim]");
  }
  auto apply = [&](Tensor& tensor) {
    const int dimension = tensor.shape[1];
    for (int head = 0; head < tensor.shape[0]; ++head) {
      for (int i = 0; i < dimension; i += 2) {
        float angle =
            position / std::pow(theta, static_cast<float>(i) / dimension);
        float cosine = std::cos(angle);
        float sine = std::sin(angle);
        float x = tensor.at2(head, i);
        float y = tensor.at2(head, i + 1);
        tensor.at2(head, i) = x * cosine - y * sine;
        tensor.at2(head, i + 1) = x * sine + y * cosine;
      }
    }
  };
  apply(q);
  apply(k);
}

// 执行两个相同形状张量的残差相加。
Tensor add(const Tensor& a, const Tensor& b) {
  RequireSameShape(a, b, "add");
  Tensor output(a.shape);
  for (size_t i = 0; i < a.size(); ++i) output.data[i] = a.data[i] + b.data[i];
  return output;
}

// 返回 logits 中最大值对应的 token id。
int argmax(const Tensor& x) {
  if (x.data.empty()) throw std::invalid_argument("argmax: empty tensor");
  return static_cast<int>(std::max_element(x.data.begin(), x.data.end()) -
                          x.data.begin());
}
}  // namespace mini_lla
