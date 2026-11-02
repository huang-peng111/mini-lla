#include "mini_llama/ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "mini_llama/matmul_dispatch.h"

namespace mini_lla {
namespace {

// 检查两个张量的形状是否一致。
void RequireSameShape(const Tensor& a, const Tensor& b, const char* caller) {
  if (a.shape != b.shape) {
    throw std::invalid_argument(std::string(caller) + ": shape mismatch " +
                                a.ShapeString() + " vs " + b.ShapeString());
  }
}

// 对逐行存储的 head 向量做旋转,配对方式为 (i, i + 1)。
void ApplyRopeNormal(Tensor& x, int position, float theta) {
  const int head_dim = x.shape[1];
  for (int head = 0; head < x.shape[0]; ++head) {
    for (int i = 0; i < head_dim; i += 2) {
      const float angle =
          static_cast<float>(position) /
          std::pow(theta, static_cast<float>(i) / static_cast<float>(head_dim));
      const float cosine = std::cos(angle);
      const float sine = std::sin(angle);
      const float first = x.At2(head, i);
      const float second = x.At2(head, i + 1);
      x.At2(head, i) = first * cosine - second * sine;
      x.At2(head, i + 1) = first * sine + second * cosine;
    }
  }
}

// NeoX 风格旋转:配对方式为 (i, i + head_dim / 2)。
void ApplyRopeNeoX(Tensor& x, int position, float theta) {
  const int head_dim = x.shape[1];
  const int half = head_dim / 2;
  for (int head = 0; head < x.shape[0]; ++head) {
    for (int i = 0; i < half; ++i) {
      const float angle = static_cast<float>(position) /
                          std::pow(theta, 2.0f * static_cast<float>(i) /
                                              static_cast<float>(head_dim));
      const float cosine = std::cos(angle);
      const float sine = std::sin(angle);
      const float first = x.At2(head, i);
      const float second = x.At2(head, i + half);
      x.At2(head, i) = first * cosine - second * sine;
      x.At2(head, i + half) = first * sine + second * cosine;
    }
  }
}

}  // namespace

Tensor Matmul(const Tensor& a, const Tensor& b) {
  return MatmulDispatch(a, b, GetMatmulMode());
}

Tensor Linear(const Tensor& x, const Tensor& weight) {
  return LinearDispatch(x, weight, GetMatmulMode());
}

Tensor RmsNorm(const Tensor& x, const Tensor& weight, float epsilon) {
  RequireSameShape(x, weight, "rms_norm");
  float square_sum = 0.0f;
  for (float value : x.data) {
    square_sum += value * value;
  }
  const float inverse_rms =
      1.0f / std::sqrt(square_sum / static_cast<float>(x.Size()) + epsilon);
  Tensor output(x.shape);
  for (size_t i = 0; i < x.Size(); ++i) {
    output.data[i] = x.data[i] * inverse_rms * weight.data[i];
  }
  return output;
}

Tensor Softmax(const Tensor& x) {
  if (x.Rank() != 1 || x.Size() == 0) {
    throw std::invalid_argument("softmax: expected a non-empty vector");
  }
  const float maximum = *std::max_element(x.data.begin(), x.data.end());
  Tensor output(x.shape);
  float total = 0.0f;
  for (size_t i = 0; i < x.Size(); ++i) {
    output.data[i] = std::exp(x.data[i] - maximum);
    total += output.data[i];
  }
  for (float& value : output.data) {
    value /= total;
  }
  return output;
}

Tensor Silu(const Tensor& x) {
  Tensor output(x.shape);
  for (size_t i = 0; i < x.Size(); ++i) {
    output.data[i] = x.data[i] / (1.0f + std::exp(-x.data[i]));
  }
  return output;
}

Tensor SwiGlu(const Tensor& gate, const Tensor& up) {
  RequireSameShape(gate, up, "swiglu");
  Tensor output = Silu(gate);
  for (size_t i = 0; i < output.Size(); ++i) {
    output.data[i] *= up.data[i];
  }
  return output;
}

Tensor Add(const Tensor& a, const Tensor& b) {
  RequireSameShape(a, b, "add");
  Tensor output(a.shape);
  for (size_t i = 0; i < a.Size(); ++i) {
    output.data[i] = a.data[i] + b.data[i];
  }
  return output;
}

void Rope(Tensor& q, Tensor& k, int position, float theta,
          RopeType rope_type) {
  if (q.Rank() != 2 || k.Rank() != 2 || q.shape[1] != k.shape[1] ||
      q.shape[1] % 2 != 0) {
    throw std::invalid_argument("rope: expected [heads, even_dim]");
  }
  if (position < 0 || !(theta > 0.0f)) {
    throw std::invalid_argument("rope: invalid position or theta");
  }
  if (rope_type == RopeType::kNeoX) {
    ApplyRopeNeoX(q, position, theta);
    ApplyRopeNeoX(k, position, theta);
  } else {
    ApplyRopeNormal(q, position, theta);
    ApplyRopeNormal(k, position, theta);
  }
}

int ArgMax(const Tensor& x) {
  if (x.Size() == 0) {
    throw std::invalid_argument("argmax: empty tensor");
  }
  return static_cast<int>(
      std::max_element(x.data.begin(), x.data.end()) - x.data.begin());
}

}  // namespace mini_lla
