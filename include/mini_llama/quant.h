#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_QUANT_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_QUANT_H_
#include "mini_llama/quantized_tensor.h"
namespace mini_lla {
// 将 F32 权重量化为 Q8_0。
QuantizedTensor quantize_q8_0(const Tensor& input);
// 将 F32 权重量化为 Q4_0。
QuantizedTensor quantize_q4_0(const Tensor& input);
// 将 F32 权重量化为 Q4_1。
QuantizedTensor quantize_q4_1(const Tensor& input);
// 使用量化权重执行二维线性层。
Tensor quantized_linear(const Tensor& input, const QuantizedTensor& weight);
}  // namespace mini_lla
#endif
