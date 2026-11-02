#include <string>

#include "mini_llama/chat.h"
#include "mini_llama/prompt_builder.h"
#include "mini_llama/sampler.h"
#include "test_main.h"

using namespace mini_lla;

// 覆盖贪心/温度/top-k 采样、参数校验和聊天 prompt 构造。
void TestSamplerChat() {
  Tensor logits({3});
  logits.data = {1.0f, 5.0f, 2.0f};

  SamplingParams greedy;
  Sampler sampler(greedy);
  CHECK_EQUAL(sampler.Sample(logits), 1);

  // 固定种子的随机采样可复现,且只从 top-k 候选里取值。
  SamplingParams random_params;
  random_params.temperature = 0.8f;
  random_params.top_k = 2;
  random_params.seed = 1234;
  const int first = Sampler(random_params).Sample(logits);
  const int second = Sampler(random_params).Sample(logits);
  CHECK_EQUAL(first, second);
  CHECK_TRUE(first == 1 || first == 2);

  // 温度为零时即使给了 top-k 也退化成贪心。
  SamplingParams top_only = random_params;
  top_only.temperature = 0.0f;
  CHECK_EQUAL(Sampler(top_only).Sample(logits), 1);

  // 非法参数和空 logits 都要报错。
  SamplingParams invalid;
  invalid.temperature = -1.0f;
  CHECK_THROWS(Sampler negative_temperature(invalid));
  invalid.temperature = 0.0f;
  invalid.top_k = -1;
  CHECK_THROWS(Sampler negative_top_k(invalid));
  CHECK_THROWS(Sampler(greedy).Sample(Tensor()));

  // 聊天会话记录消息,清空后不再保留历史。
  ChatSession session;
  session.AddMessage("user", "hello");
  session.AddMessage("assistant", "hi");
  CHECK_EQUAL(session.Size(), 2u);

  const std::string plain = BuildPrompt(session.Messages(), "");
  CHECK_TRUE(plain.find("User: hello") != std::string::npos);
  CHECK_TRUE(plain.find("Assistant:") != std::string::npos);

  const std::string qwen = BuildPrompt(session.Messages(), kQwen2Template);
  CHECK_TRUE(qwen.find("<|im_start|>user\nhello<|im_end|>") !=
             std::string::npos);
  CHECK_TRUE(qwen.find("<|im_start|>assistant\n") != std::string::npos);
  CHECK_TRUE(qwen.find("You are a helpful assistant.") != std::string::npos);

  session.Clear();
  CHECK_EQUAL(session.Size(), 0u);
}
