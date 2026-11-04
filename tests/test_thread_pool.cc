#include <atomic>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "mini_llama/matmul_dispatch.h"
#include "mini_llama/ops.h"
#include "mini_llama/thread_pool.h"
#include "test_main.h"

using namespace mini_lla;

// 覆盖线程池的分块、异常传播、重复调用和数值一致性。
void TestThreadPool() {
  CHECK_TRUE(ThreadCount() >= 1);

  // 每个下标必须被处理且只被处理一次。
  const int total = 1000;
  std::vector<int> hits(static_cast<size_t>(total), 0);
  ParallelFor(total, [&](int begin, int end) {
    for (int i = begin; i < end; ++i) hits[static_cast<size_t>(i)] += 1;
  });
  for (int value : hits) CHECK_EQUAL(value, 1);

  // 连续多次调用不应该死锁。
  for (int repeat = 0; repeat < 20; ++repeat) {
    std::atomic<int> sum{0};
    ParallelFor(500, [&](int begin, int end) { sum += end - begin; });
    CHECK_EQUAL(sum.load(), 500);
  }

  // 任务里的异常要传回调用线程。
  CHECK_THROWS(ParallelFor(10000, [](int, int) {
    throw std::runtime_error("boom");
  }));

  // 空任务直接返回。
  int calls = 0;
  ParallelFor(0, [&](int, int) { ++calls; });
  CHECK_EQUAL(calls, 0);

  // 单线程与多线程的线性层结果必须一致。
  Tensor x({64});
  for (int i = 0; i < 64; ++i) x.data[i] = std::sin(0.1f * i);
  Tensor weight({128, 64});
  for (size_t i = 0; i < weight.Size(); ++i) {
    weight.data[i] = std::cos(0.05f * static_cast<float>(i));
  }
  SetThreadCount(1);
  const Tensor serial = Linear(x, weight);
  SetThreadCount(4);
  const Tensor parallel = Linear(x, weight);
  for (size_t i = 0; i < serial.Size(); ++i) {
    CHECK_NEAR(serial.data[i], parallel.data[i], 1e-5);
  }
  SetThreadCount(0);
}
