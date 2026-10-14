#include "mini_llama/sampler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mini_llama/ops.h"

namespace mini_lla {

Sampler::Sampler(const SamplingParams& params) : params_(params) {
  if (params_.temperature < 0.0f || params_.top_k < 0) {
    throw std::invalid_argument("sampling parameters must be non-negative");
  }
  generator_.seed(params_.seed == 0 ? std::random_device{}() : params_.seed);
}

int Sampler::sample(const Tensor& logits) {
  if (logits.data.empty()) throw std::invalid_argument("empty logits");
  if (params_.temperature == 0.0f && params_.top_k == 0) {
    return argmax(logits);
  }

  // 先按 logit 排序，再截取 top-k 候选，减少随机采样范围。
  std::vector<std::pair<float, int>> candidates;
  for (size_t i = 0; i < logits.size(); ++i) {
    candidates.emplace_back(logits.data[i], static_cast<int>(i));
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& left, const auto& right) {
              return left.first > right.first;
            });
  if (params_.top_k > 0 && params_.top_k < static_cast<int>(candidates.size())) {
    candidates.resize(static_cast<size_t>(params_.top_k));
  }
  if (params_.temperature == 0.0f) return candidates.front().second;

  float maximum = candidates.front().first;
  std::vector<float> probabilities(candidates.size());
  float total = 0.0f;
  for (size_t i = 0; i < candidates.size(); ++i) {
    probabilities[i] = std::exp((candidates[i].first - maximum) /
                                 params_.temperature);
    total += probabilities[i];
  }
  std::uniform_real_distribution<float> distribution(0.0f, total);
  const float target = distribution(generator_);
  float cumulative = 0.0f;
  for (size_t i = 0; i < probabilities.size(); ++i) {
    cumulative += probabilities[i];
    if (target <= cumulative) return candidates[i].second;
  }
  return candidates.back().second;
}

}  // namespace mini_lla
