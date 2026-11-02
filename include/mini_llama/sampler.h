#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_SAMPLER_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_SAMPLER_H_

#include <random>

#include "mini_llama/tensor.h"

namespace mini_lla {

// 采样参数:温度、top-k 候选数和随机种子。
struct SamplingParams {
  float temperature = 0.0f;  // 0 表示贪心
  int top_k = 0;             // 0 表示不截断候选
  unsigned int seed = 0;     // 0 表示每次使用随机种子
};

// 根据 logits 选择下一个 token,支持贪心、温度和 top-k 采样。
class Sampler {
 public:
  // 使用给定参数创建采样器,并初始化随机数发生器。
  explicit Sampler(const SamplingParams& params);

  // 从 logits 中采样一个 token id。
  int Sample(const Tensor& logits);

 private:
  // 对候选集合做多项式采样,temperature 为 0 时取最大值。
  int SampleFromCandidates(
      const std::vector<std::pair<float, int>>& candidates);

  SamplingParams params_;
  std::mt19937 generator_;
};

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_SAMPLER_H_
