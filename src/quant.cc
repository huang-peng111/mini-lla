#include "mini_llama/quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "mini_llama/ops.h"

namespace mini_lla {
namespace {

// 把 F32 转成 FP16 位模式,量化 block 的系数按 FP16 保存。
uint16_t FloatToFp16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffffu;

  if (exponent <= 0) {
    if (exponent < -10) return static_cast<uint16_t>(sign);
    mantissa |= 0x800000u;
    uint32_t shifted = mantissa >> (1 - exponent);
    if ((shifted & 0x00001000u) != 0) shifted += 0x00002000u;
    return static_cast<uint16_t>(sign | (shifted >> 13));
  }
  if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  if ((mantissa & 0x00001000u) != 0) {
    mantissa += 0x00002000u;
    if ((mantissa & 0x00800000u) != 0) {
      mantissa = 0;
      ++exponent;
      if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                               (mantissa >> 13));
}

// 把 FP16 位模式还原成 F32。
float Fp16ToFloat(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
  uint32_t exponent = (value >> 10) & 0x1fu;
  uint32_t mantissa = value & 0x03ffu;
  uint32_t bits = 0;

  if (exponent == 0) {
    if (mantissa != 0) {
      exponent = 1;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03ffu;
      bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    } else {
      bits = sign;
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
  }

  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

// 四舍五入取整并限制到 [low, high]。
int ClampToInt(float value, int low, int high) {
  const int rounded = static_cast<int>(std::lround(value));
  return std::max(low, std::min(high, rounded));
}

// 返回 [values, values + count) 中绝对值最大的元素。
float MaxAbs(const float* values, int count) {
  float maximum = 0.0f;
  for (int i = 0; i < count; ++i) {
    maximum = std::max(maximum, std::fabs(values[i]));
  }
  return maximum;
}

// 量化一个 Q8_0 block:scale = max|x| / 127。
BlockQ8_0 QuantizeBlockQ80(const float* values, int count) {
  BlockQ8_0 block{};
  const float maximum = MaxAbs(values, count);
  if (maximum == 0.0f) return block;

  block.d = FloatToFp16(maximum / 127.0f);
  const float scale = Fp16ToFloat(block.d);
  for (int i = 0; i < count; ++i) {
    block.qs[i] = static_cast<int8_t>(ClampToInt(values[i] / scale, -127, 127));
  }
  return block;
}

// 量化一个 Q4_0 block:scale = max|x| / 8,量化值加上 8 后落在 [0, 15]。
BlockQ4_0 QuantizeBlockQ40(const float* values, int count) {
  BlockQ4_0 block{};
  const float maximum = MaxAbs(values, count);
  if (maximum == 0.0f) return block;

  block.d = FloatToFp16(maximum / 8.0f);
  const float scale = Fp16ToFloat(block.d);
  for (int i = 0; i < count; ++i) {
    const int quantized = ClampToInt(values[i] / scale, -8, 7) + 8;
    if (i < kBlockSize / 2) {
      block.qs[i] = static_cast<uint8_t>(quantized);
    } else {
      block.qs[i - kBlockSize / 2] |= static_cast<uint8_t>(quantized << 4);
    }
  }
  return block;
}

// 量化一个 Q4_1 block:scale = (max - min) / 15,量化值落在 [0, 15]。
BlockQ4_1 QuantizeBlockQ41(const float* values, int count) {
  BlockQ4_1 block{};
  float minimum = values[0];
  float maximum = values[0];
  for (int i = 1; i < count; ++i) {
    minimum = std::min(minimum, values[i]);
    maximum = std::max(maximum, values[i]);
  }

  block.m = FloatToFp16(minimum);
  if (maximum == minimum) return block;

  block.d = FloatToFp16((maximum - minimum) / 15.0f);
  const float scale = Fp16ToFloat(block.d);
  for (int i = 0; i < count; ++i) {
    const int quantized = ClampToInt((values[i] - minimum) / scale, 0, 15);
    if (i < kBlockSize / 2) {
      block.qs[i] = static_cast<uint8_t>(quantized);
    } else {
      block.qs[i - kBlockSize / 2] |= static_cast<uint8_t>(quantized << 4);
    }
  }
  return block;
}

// 取回 block 中第 index 个元素的 F32 值,index 相对 block 起点。
float DequantizeElement(const BlockQ8_0& block, int index) {
  return Fp16ToFloat(block.d) * static_cast<float>(block.qs[index]);
}

// Q4_0 的前半元素在低半字节,后半元素在高半字节,都要减去 8。
float DequantizeElement(const BlockQ4_0& block, int index) {
  const int packed = index < kBlockSize / 2
                         ? block.qs[index] & 0x0f
                         : block.qs[index - kBlockSize / 2] >> 4;
  return Fp16ToFloat(block.d) * static_cast<float>(packed - 8);
}

// Q4_1 的量化值直接落在 [0, 15],再乘 scale 加上最小值。
float DequantizeElement(const BlockQ4_1& block, int index) {
  const int packed = index < kBlockSize / 2
                         ? block.qs[index] & 0x0f
                         : block.qs[index - kBlockSize / 2] >> 4;
  return Fp16ToFloat(block.d) * static_cast<float>(packed) +
         Fp16ToFloat(block.m);
}

// 计算输入向量与量化权重某一行的点积,block_offset 是该行第一个 block。
template <typename Block>
float DotQuantizedRow(const std::vector<Block>& blocks, size_t block_offset,
                      const float* input, int cols) {
  const int block_count = (cols + kBlockSize - 1) / kBlockSize;
  float sum = 0.0f;
  for (int b = 0; b < block_count; ++b) {
    const Block& block = blocks[block_offset + static_cast<size_t>(b)];
    const int begin = b * kBlockSize;
    const int end = std::min(begin + kBlockSize, cols);
    for (int i = begin; i < end; ++i) {
      sum += DequantizeElement(block, i - begin) * input[i];
    }
  }
  return sum;
}

// 把量化权重第 row 行反量化写入长度为 cols 的 output。
template <typename Block>
void DequantizeRowHelper(const std::vector<Block>& blocks, size_t block_offset,
                         int cols, float* output) {
  const int block_count = (cols + kBlockSize - 1) / kBlockSize;
  for (int b = 0; b < block_count; ++b) {
    const Block& block = blocks[block_offset + static_cast<size_t>(b)];
    const int begin = b * kBlockSize;
    const int end = std::min(begin + kBlockSize, cols);
    for (int i = begin; i < end; ++i) {
      output[i] = DequantizeElement(block, i - begin);
    }
  }
}

// 用 block 权重执行线性层:一维输入返回一维输出,二维输入逐行计算。
template <typename Block>
Tensor LinearWithBlocks(const Tensor& input, const std::vector<Block>& blocks,
                        int rows, int cols) {
  const bool is_vector = input.Rank() == 1;
  const int input_rows = is_vector ? 1 : input.shape[0];
  if (static_cast<int>(input.data.size()) != input_rows * cols) {
    throw std::invalid_argument("quantized linear: input size mismatch");
  }

  Tensor output(is_vector ? std::vector<int>{rows}
                          : std::vector<int>{input_rows, rows});
  const int blocks_per_row = (cols + kBlockSize - 1) / kBlockSize;
  for (int r = 0; r < input_rows; ++r) {
    const float* source = input.data.data() + static_cast<size_t>(r) * cols;
    float* destination = output.data.data() + static_cast<size_t>(r) * rows;
    for (int j = 0; j < rows; ++j) {
      destination[j] = DotQuantizedRow(
          blocks, static_cast<size_t>(j) * blocks_per_row, source, cols);
    }
  }
  return output;
}

}  // namespace

QuantizedTensor Quantize(const Tensor& source, QuantType type) {
  if (source.Rank() != 2) {
    throw std::invalid_argument("Quantize expects a 2D tensor");
  }
  if (type == QuantType::kF32) return MakeF32Weight(source);

  const int rows = source.shape[0];
  const int cols = source.shape[1];
  const int blocks_per_row = (cols + kBlockSize - 1) / kBlockSize;

  QuantizedTensor result;
  result.type = type;
  result.rows = rows;
  result.cols = cols;
  for (int row = 0; row < rows; ++row) {
    const float* values = source.data.data() + static_cast<size_t>(row) * cols;
    for (int b = 0; b < blocks_per_row; ++b) {
      const int begin = b * kBlockSize;
      const int count = std::min(kBlockSize, cols - begin);
      const float* block_values = values + begin;
      switch (type) {
        case QuantType::kQ8_0:
          result.q8_0.push_back(QuantizeBlockQ80(block_values, count));
          break;
        case QuantType::kQ4_0:
          result.q4_0.push_back(QuantizeBlockQ40(block_values, count));
          break;
        case QuantType::kQ4_1:
          result.q4_1.push_back(QuantizeBlockQ41(block_values, count));
          break;
        case QuantType::kF32:
          break;
      }
    }
  }
  return result;
}

Tensor Dequantize(const QuantizedTensor& weight) {
  Tensor output({weight.rows, weight.cols});
  for (int row = 0; row < weight.rows; ++row) {
    float* destination =
        output.data.data() + static_cast<size_t>(row) * weight.cols;
    DequantizeRow(weight, row, destination);
  }
  return output;
}

void DequantizeRow(const QuantizedTensor& weight, int row, float* output) {
  if (row < 0 || row >= weight.rows) {
    throw std::out_of_range("DequantizeRow: row out of range");
  }
  if (weight.IsF32()) {
    const float* source =
        weight.f32.data.data() + static_cast<size_t>(row) * weight.cols;
    std::copy_n(source, weight.cols, output);
    return;
  }

  const size_t offset =
      static_cast<size_t>(row) * static_cast<size_t>(weight.BlocksPerRow());
  switch (weight.type) {
    case QuantType::kQ8_0:
      DequantizeRowHelper(weight.q8_0, offset, weight.cols, output);
      break;
    case QuantType::kQ4_0:
      DequantizeRowHelper(weight.q4_0, offset, weight.cols, output);
      break;
    case QuantType::kQ4_1:
      DequantizeRowHelper(weight.q4_1, offset, weight.cols, output);
      break;
    case QuantType::kF32:
      break;
  }
}

Tensor Linear(const Tensor& input, const QuantizedTensor& weight) {
  switch (weight.type) {
    case QuantType::kF32:
      return mini_lla::Linear(input, weight.f32);
    case QuantType::kQ8_0:
      return LinearWithBlocks(input, weight.q8_0, weight.rows, weight.cols);
    case QuantType::kQ4_0:
      return LinearWithBlocks(input, weight.q4_0, weight.rows, weight.cols);
    case QuantType::kQ4_1:
      return LinearWithBlocks(input, weight.q4_1, weight.rows, weight.cols);
  }
  throw std::invalid_argument("Linear: unknown quant type");
}

float LinearError(const Tensor& input, const Tensor& weight, QuantType type) {
  const Tensor reference = mini_lla::Linear(input, weight);
  const Tensor quantized = Linear(input, Quantize(weight, type));
  if (reference.Size() != quantized.Size()) {
    throw std::runtime_error("LinearError: output size mismatch");
  }
  float worst = 0.0f;
  for (size_t i = 0; i < reference.Size(); ++i) {
    worst = std::max(worst, std::fabs(reference.data[i] - quantized.data[i]));
  }
  return worst;
}

}  // namespace mini_lla
