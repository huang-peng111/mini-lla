#include <stdexcept>

#include "mini_llama/tensor.h"
#include "test_main.h"

using mini_lla::Tensor;

// 覆盖 Tensor 的构造、索引、reshape 和错误路径。
void TestTensor() {
  Tensor tensor({2, 3});
  CHECK_EQUAL(tensor.Size(), 6);
  CHECK_EQUAL(tensor.Rank(), 2);

  tensor.At2(1, 2) = 5.0f;
  CHECK_NEAR(tensor.At2(1, 2), 5.0f, 1e-6);
  CHECK_EQUAL(tensor.Index({1, 1}), 4);
  CHECK_NEAR(*tensor.RowPtr(1), 0.0f, 1e-6);
  CHECK_NEAR(tensor.RowPtr(1)[2], 5.0f, 1e-6);

  const Tensor reshaped = tensor.Reshape({3, 2});
  CHECK_EQUAL(reshaped.Size(), 6);
  CHECK_EQUAL(reshaped.shape[0], 3);
  CHECK_NEAR(reshaped.At2(2, 1), 5.0f, 1e-6);

  CHECK_TRUE(tensor.ShapeString() == "[2, 3]");
  tensor.RequireShape({2, 3}, "test");
  CHECK_THROWS(tensor.RequireShape({3, 2}, "test"));

  // 非法形状、越界访问和非法 reshape 都要报错。
  CHECK_THROWS(Tensor({0, 3}));
  CHECK_THROWS(Tensor({-1}));
  CHECK_THROWS(tensor.At2(2, 0));
  CHECK_THROWS(tensor.At1(0));
  CHECK_THROWS(tensor.Reshape({4, 2}));
  CHECK_THROWS(tensor.Index({1}));
}
