#include "mini_llama/chat.h"
#include "mini_llama/sampler.h"

#include <stdexcept>

using namespace mini_lla;

// 验证 greedy、top-k 采样和多轮 prompt 的最小行为。
int test_sampler_chat() {
  Tensor logits({3});
  logits.data = {1.0f, 5.0f, 2.0f};
  SamplingParams greedy_params;
  Sampler greedy(greedy_params);
  if (greedy.sample(logits) != 1) return 1;

  SamplingParams top_k_params;
  top_k_params.temperature = 0.5f;
  top_k_params.top_k = 2;
  top_k_params.seed = 42;
  Sampler top_k(top_k_params);
  const int sampled = top_k.sample(logits);
  if (sampled != 1 && sampled != 2) return 1;

  ChatSession session;
  session.add_message("user", "hello");
  session.add_message("assistant", "hi");
  if (session.build_prompt().find("<|user|>") == std::string::npos) return 1;
  session.clear();
  if (session.size() != 0) return 1;
  return 0;
}
