#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_QUANT_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_QUANT_H_

#include <vector>

#include "mini_llama/quantized_tensor.h"

namespace mini_lla {

// 把二维 F32 权重量化成指定格式;type 为 kF32 时直接复制。
QuantizedTensor Quantize(const Tensor& source, QuantType type);

// 把量化权重还原成完整的 F32 张量。
Tensor Dequantize(const QuantizedTensor& weight);

// 只反量化第 row 行写入 output(embedding 查表用,避免复制整个权重)。
// output 必须至少有 cols 个元素。
void DequantizeRow(const QuantizedTensor& weight, int row, float* output);

// 用 F32 输入执行量化线性层,输入形状为 [cols] 或 [行数, cols]。
Tensor Linear(const Tensor& input, const QuantizedTensor& weight);

// 比较 F32 线性层与其量化版本的输出,返回最大绝对误差。
float LinearError(const Tensor& input, const Tensor& weight, QuantType type);

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_QUANT_H_
