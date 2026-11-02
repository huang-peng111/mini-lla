#include <cmath>
#include <vector>

#include "mini_llama/ops.h"
#include "mini_llama/quant.h"
#include "test_main.h"

using namespace mini_lla;

// 覆盖三种量化格式的 block 布局、量化误差和量化线性层。
void TestQuant() {
  // block 布局必须与 ggml 一致。
  CHECK_EQUAL(sizeof(BlockQ8_0), 34u);
  CHECK_EQUAL(sizeof(BlockQ4_0), 18u);
  CHECK_EQUAL(sizeof(BlockQ4_1), 20u);

  // 第一行是负值绝对值更大的不对称数据,第二行是 0 到 1 的正值数据。
  Tensor weight({2, 32});
  for (int i = 0; i < 32; ++i) {
    weight.data[i] = 0.01f * static_cast<float>(i) - 0.25f;
    weight.data[32 + i] = 0.01f * static_cast<float>(i);
  }

  const QuantType types[] = {QuantType::kQ8_0, QuantType::kQ4_0,
                             QuantType::kQ4_1};
  for (QuantType type : types) {
    const QuantizedTensor quantized = Quantize(weight, type);
    CHECK_EQUAL(quantized.rows, 2);
    CHECK_EQUAL(quantized.cols, 32);
    CHECK_EQUAL(quantized.BlocksPerRow(), 1);

    const Tensor restored = Dequantize(quantized);
    CHECK_TRUE(restored.shape == weight.shape);
    const float tolerance = type == QuantType::kQ8_0 ? 0.01f : 0.08f;
    for (size_t i = 0; i < weight.Size(); ++i) {
      CHECK_TRUE(std::isfinite(restored.data[i]));
      CHECK_TRUE(std::fabs(restored.data[i] - weight.data[i]) <= tolerance);
    }
    CHECK_TRUE(QuantizedBytes(quantized) < weight.Size() * sizeof(float));
  }

  // Q8_0 的线性层误差要明显小于 Q4_0。
  Tensor input({32});
  for (int i = 0; i < 32; ++i) input.data[i] = 0.05f * (i % 5);
  CHECK_TRUE(LinearError(input, weight, QuantType::kQ8_0) < 0.05f);
  CHECK_TRUE(LinearError(input, weight, QuantType::kQ4_0) < 0.3f);

  // 列数不是 32 的整数倍时,尾部 block 也要正确往返。
  Tensor tail({1, 40});
  for (int i = 0; i < 40; ++i) tail.data[i] = std::sin(static_cast<float>(i));
  const QuantizedTensor tail_quantized = Quantize(tail, QuantType::kQ8_0);
  CHECK_EQUAL(tail_quantized.BlocksPerRow(), 2);
  const Tensor tail_restored = Dequantize(tail_quantized);
  for (size_t i = 0; i < tail.Size(); ++i) {
    CHECK_NEAR(tail_restored.data[i], tail.data[i], 0.02);
  }

  // 取回单行用于 embedding 查表。
  std::vector<float> row(32, 0.0f);
  DequantizeRow(Quantize(weight, QuantType::kQ8_0), 1, row.data());
  for (int i = 0; i < 32; ++i) {
    CHECK_NEAR(row[static_cast<size_t>(i)], weight.data[32 + i], 0.02);
  }
  CHECK_THROWS(DequantizeRow(Quantize(weight, QuantType::kQ8_0), 5, row.data()));

  // 量化线性层:一维输入返回一维,二维输入逐行计算。
  Tensor small({4, 32});
  for (size_t i = 0; i < small.Size(); ++i) {
    small.data[i] = 0.1f * static_cast<float>(i % 7) - 0.3f;
  }
  const QuantizedTensor small_quantized = Quantize(small, QuantType::kQ8_0);
  const Tensor vector_output = Linear(input, small_quantized);
  CHECK_EQUAL(vector_output.Rank(), 1);
  CHECK_EQUAL(vector_output.Size(), 4u);

  Tensor rows_input({3, 32});
  rows_input.data.assign(96, 0.5f);
  const Tensor rows_output = Linear(rows_input, small_quantized);
  CHECK_EQUAL(rows_output.Rank(), 2);
  CHECK_EQUAL(rows_output.shape[0], 3);
  CHECK_EQUAL(rows_output.shape[1], 4);
  CHECK_NEAR(rows_output.data[0], rows_output.data[4], 1e-5);

  // F32 权重走同一条接口,结果与直接线性层一致。
  const QuantizedTensor f32_weight = MakeF32Weight(weight);
  CHECK_TRUE(f32_weight.IsF32());
  CHECK_NEAR(Linear(input, f32_weight).data[0], Linear(input, weight).data[0],
             1e-6);

  // 形状不匹配时报错。
  CHECK_THROWS(Linear(Tensor({31}), small_quantized));
  CHECK_THROWS(Quantize(Tensor({4}), QuantType::kQ8_0));
}
