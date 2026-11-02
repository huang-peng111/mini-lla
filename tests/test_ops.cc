#include <cmath>

#include "mini_llama/matmul_dispatch.h"
#include "mini_llama/ops.h"
#include "test_main.h"

using namespace mini_lla;

// 覆盖矩阵乘、归一化、softmax、激活函数、RoPE 和 argmax。
void TestOps() {
  Tensor a({2, 3});
  a.data = {1, 2, 3, 4, 5, 6};
  Tensor b({3, 2});
  b.data = {1, 2, 3, 4, 5, 6};

  const Tensor optimized = Matmul(a, b);
  const Tensor naive = MatmulDispatch(a, b, MatmulMode::kNaive);
  CHECK_NEAR(optimized.At2(0, 0), 22.0f, 1e-4);
  CHECK_NEAR(optimized.At2(1, 1), 64.0f, 1e-4);
  for (size_t i = 0; i < optimized.Size(); ++i) {
    CHECK_NEAR(optimized.data[i], naive.data[i], 1e-4);
  }

  // 线性层支持一维输入,并按权重行数决定输出长度。
  Tensor weight({2, 3});
  weight.data = {1, 0, 0, 0, 1, 0};
  Tensor row({3});
  row.data = {1, 2, 3};
  const Tensor projected = Linear(row, weight);
  CHECK_EQUAL(projected.Rank(), 1);
  CHECK_EQUAL(projected.Size(), 2u);
  CHECK_NEAR(projected.data[0], 1.0f, 1e-5);
  CHECK_NEAR(projected.data[1], 2.0f, 1e-5);

  Tensor x({3});
  x.data = {1, 2, 3};
  Tensor norm_weight({3});
  norm_weight.data = {1, 1, 1};
  CHECK_NEAR(RmsNorm(x, norm_weight, 1e-5f).data[0], 0.46291f, 1e-4);
  CHECK_THROWS(RmsNorm(x, Tensor({2}), 1e-5f));

  const Tensor probabilities = Softmax(x);
  CHECK_NEAR(probabilities.data[0] + probabilities.data[1] + probabilities.data[2],
             1.0f, 1e-5);
  CHECK_TRUE(probabilities.data[2] > probabilities.data[1]);
  CHECK_THROWS(Softmax(Tensor({2, 2})));

  Tensor gate({2});
  gate.data = {0.0f, 1.0f};
  Tensor up({2});
  up.data = {2.0f, 2.0f};
  const Tensor glu = SwiGlu(gate, up);
  CHECK_NEAR(glu.data[0], 0.0f, 1e-6);
  // Silu(1) * up = 2 / (1 + e^-1)。
  CHECK_NEAR(glu.data[1], 2.0f / (1.0f + std::exp(-1.0f)), 1e-5);
  CHECK_THROWS(SwiGlu(gate, Tensor({3})));

  Tensor sum = Add(gate, up);
  CHECK_NEAR(sum.data[0], 2.0f, 1e-6);
  CHECK_THROWS(Add(gate, Tensor({3})));

  CHECK_EQUAL(ArgMax(x), 2);
  CHECK_THROWS(ArgMax(Tensor({0})));

  // RoPE:位置 1 且 head_dim 为 2 时正好旋转 1 弧度。
  Tensor q({1, 2});
  q.data = {1.0f, 0.0f};
  Tensor k({1, 2});
  k.data = {1.0f, 0.0f};
  const RopeType rope_types[] = {RopeType::kNormal, RopeType::kNeoX};
  for (RopeType type : rope_types) {
    Tensor rotated_q = q;
    Tensor rotated_k = k;
    Rope(rotated_q, rotated_k, 1, 10000.0f, type);
    CHECK_NEAR(rotated_q.data[0], std::cos(1.0f), 1e-5);
    CHECK_NEAR(rotated_q.data[1], std::sin(1.0f), 1e-5);
    CHECK_NEAR(rotated_k.data[1], std::sin(1.0f), 1e-5);
  }

  Tensor odd({1, 3});
  CHECK_THROWS(Rope(odd, odd, 1, 10000.0f, RopeType::kNormal));
  CHECK_THROWS(Rope(q, k, -1, 10000.0f, RopeType::kNormal));
}
