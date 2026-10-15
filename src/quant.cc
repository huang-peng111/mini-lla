#include "mini_llama/quant.h"
#include "mini_llama/ops.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace mini_lla {
namespace {
// 按 32 个元素一个 block 计算统一 scale 和 zero point。
QuantizedTensor quantize(const Tensor& input, QuantType type, int limit) {
  if (input.shape.size() != 2) throw std::invalid_argument("quantization expects a matrix");
  QuantizedTensor result; result.type = type; result.shape = input.shape; result.f32 = input;
  const int block = 32; const size_t blocks = (input.size() + block - 1) / block;
  result.scales.resize(blocks); result.zeros.resize(blocks); result.values.resize(input.size());
  for (size_t b = 0; b < blocks; ++b) { const size_t begin = b * block; const size_t end = std::min(begin + block, input.size()); float maximum = 0.0f; float minimum = 0.0f; for (size_t i = begin; i < end; ++i) { maximum = std::max(maximum, input.data[i]); minimum = std::min(minimum, input.data[i]); } const float zero = type == QuantType::kQ4_1 ? minimum : 0.0f; const float scale = std::max((maximum - zero) / limit, 1e-8f); result.scales[b] = scale; result.zeros[b] = zero; for (size_t i = begin; i < end; ++i) result.values[i] = static_cast<int8_t>(std::round((input.data[i] - zero) / scale)); }
  return result;
}
}
// 使用 8 位有符号值保存量化结果。
QuantizedTensor quantize_q8_0(const Tensor& input) { return quantize(input, QuantType::kQ8_0, 127); }
// 使用 4 位范围保存对称量化结果。
QuantizedTensor quantize_q4_0(const Tensor& input) { return quantize(input, QuantType::kQ4_0, 15); }
// 使用 4 位范围和 zero point 保存非对称量化结果。
QuantizedTensor quantize_q4_1(const Tensor& input) { return quantize(input, QuantType::kQ4_1, 15); }
// 将 block 量化数据还原为 F32 张量。
Tensor QuantizedTensor::dequantize() const { if (type == QuantType::kF32) return f32; Tensor output(shape); const int block = 32; for (size_t i = 0; i < output.size(); ++i) output.data[i] = values[i] * scales[i / block] + zeros[i / block]; return output; }
// 量化权重先还原，再复用基础线性层完成 CPU 计算。
Tensor quantized_linear(const Tensor& input, const QuantizedTensor& weight) { return linear(input, weight.dequantize()); }
}  // namespace mini_lla
