#include "mini_llama/benchmark.h"
#include "mini_llama/loader.h"
#include "test_main.h"

using namespace mini_lla;

// 覆盖 benchmark 的 token 计数与计时字段。
void TestBenchmark() {
  const LoadedModel loaded = LoadModelAndTokenizer(test::TinyModelDir());
  SamplingParams params;
  const BenchmarkResult result = RunBenchmark(loaded.model, *loaded.tokenizer,
                                              "hello", 3, params, false);
  CHECK_EQUAL(result.prompt_tokens, 6);  // ASCII tokenizer 会在开头加 BOS
  CHECK_TRUE(result.generated_tokens <= 3);
  CHECK_TRUE(result.decode_tokens <= result.generated_tokens);
  CHECK_TRUE(result.prefill_ms >= 0.0);
  CHECK_TRUE(result.decode_ms >= 0.0);
  CHECK_TRUE(result.TokensPerSecond() >= 0.0);
  CHECK_TRUE(result.DecodeTokensPerSecond() >= 0.0);

  // 空 prompt 与超长 prompt 都要报错。
  CHECK_THROWS(
      RunBenchmark(loaded.model, *loaded.tokenizer, "", 1, params, false));
  CHECK_THROWS(RunBenchmark(loaded.model, *loaded.tokenizer,
                            std::string(400, 'a'), 1, params, false));
}
