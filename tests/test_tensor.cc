#include "mini_llama/tensor.h"
using namespace mini_lla;
// 验证 Tensor 创建、访问和 reshape 的核心行为。
int test_tensor() {
  Tensor t({2, 3});
  t.at2(1, 2) = 5;
  if (t.size() != 6 || t.at2(1, 2) != 5) return 1;
  if (t.reshape({3, 2}).size() != 6) return 1;
  return 0;
}
