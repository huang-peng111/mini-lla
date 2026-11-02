#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_QUANTIZED_TENSOR_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_QUANTIZED_TENSOR_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "mini_llama/tensor.h"

namespace mini_lla {

// 权重存储格式;kF32 表示未压缩的 F32 权重。
enum class QuantType { kF32, kQ8_0, kQ4_0, kQ4_1 };

// 每个量化 block 覆盖 32 个连续元素,与 ggml/llama.cpp 的块布局一致。
constexpr int kBlockSize = 32;

// Q8_0 block:FP16 缩放系数加 32 个 int8 量化值。
struct BlockQ8_0 {
  uint16_t d;              // FP16 缩放系数
  int8_t qs[kBlockSize];   // 量化值
};

// Q4_0 block:FP16 缩放系数加 32 个 4bit 量化值(每字节两个)。
struct BlockQ4_0 {
  uint16_t d;                     // FP16 缩放系数
  uint8_t qs[kBlockSize / 2];     // 低半字节存前半,高半字节存后半
};

// Q4_1 block:FP16 缩放系数和最小值加 32 个 4bit 量化值。
struct BlockQ4_1 {
  uint16_t d;                     // FP16 缩放系数
  uint16_t m;                     // FP16 最小值
  uint8_t qs[kBlockSize / 2];     // 4bit 量化值
};

static_assert(sizeof(BlockQ8_0) == 34, "Q8_0 block must be 34 bytes");
static_assert(sizeof(BlockQ4_0) == 18, "Q4_0 block must be 18 bytes");
static_assert(sizeof(BlockQ4_1) == 20, "Q4_1 block must be 20 bytes");

// 二维权重矩阵,可以按 F32 或任意一种量化 block 存储。
// 只有 type 对应的那个字段有效,其余字段为空。
struct QuantizedTensor {
  QuantType type = QuantType::kF32;
  int rows = 0;  // 输出维度
  int cols = 0;  // 输入维度
  Tensor f32;                      // type == kF32 时有效
  std::vector<BlockQ8_0> q8_0;     // type == kQ8_0 时有效
  std::vector<BlockQ4_0> q4_0;     // type == kQ4_0 时有效
  std::vector<BlockQ4_1> q4_1;     // type == kQ4_1 时有效

  // 元素总数 rows * cols。
  size_t Size() const {
    return static_cast<size_t>(rows) * static_cast<size_t>(cols);
  }

  // 每一行占用的 block 数(列数不是 32 的整数倍时向上取整)。
  int BlocksPerRow() const { return (cols + kBlockSize - 1) / kBlockSize; }

  // 是否为未量化的 F32 权重。
  bool IsF32() const { return type == QuantType::kF32; }
};

// 返回格式名称,用于日志和统计,例如 "Q8_0"。
const char* QuantTypeName(QuantType type);

// 把二维 F32 张量包装成 kF32 权重。
QuantizedTensor MakeF32Weight(Tensor tensor);

// 返回权重实际占用的字节数。
size_t QuantizedBytes(const QuantizedTensor& weight);

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_QUANTIZED_TENSOR_H_
