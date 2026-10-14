#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_SAMPLER_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_SAMPLER_H_

#include <random>

#include "mini_llama/tensor.h"

namespace mini_lla {

// 描述生成时的温度、候选数量和随机种子。
struct SamplingParams {
  float temperature = 0.0f;
  int top_k = 0;
  unsigned int seed = 0;
};

// 根据 logits 选择下一个 token，温度为零时使用 greedy。
class Sampler {
 public:
  // 使用指定参数创建采样器。
  explicit Sampler(const SamplingParams& params);
  // 按参数从 logits 中选择 token id。
  int sample(const Tensor& logits);

 private:
  SamplingParams params_;
  std::mt19937 generator_;
};

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_SAMPLER_H_
