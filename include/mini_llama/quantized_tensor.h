#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_QUANTIZED_TENSOR_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_QUANTIZED_TENSOR_H_
#include <vector>
#include "mini_llama/tensor.h"
namespace mini_lla {
// 当前支持的量化格式。
enum class QuantType { kF32, kQ8_0, kQ4_0, kQ4_1 };
// 保存量化权重及其原始二维形状。
struct QuantizedTensor {
  QuantType type = QuantType::kF32;
  std::vector<int> shape;
  std::vector<float> scales;
  std::vector<float> zeros;
  std::vector<int8_t> values;
  Tensor f32;
  // 将量化张量还原为 F32。
  Tensor dequantize() const;
};
}  // namespace mini_lla
#endif
