#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_BENCHMARK_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_BENCHMARK_H_

#include <string>

#include "mini_llama/model.h"
#include "mini_llama/sampler.h"
#include "mini_llama/tokenizer.h"

namespace mini_lla {

// 一次 benchmark 的统计结果。
struct BenchmarkResult {
  int prompt_tokens = 0;
  int generated_tokens = 0;
  int decode_tokens = 0;
  double prefill_ms = 0.0;
  double decode_ms = 0.0;

  // 总吞吐:生成 token 数除以总耗时。
  double TokensPerSecond() const;

  // 解码吞吐:decode 阶段的 token/s。
  double DecodeTokensPerSecond() const;
};

// 先 prefill 整个 prompt,再逐个生成 token,返回耗时统计。
// verbose 为 true 时打印每一步的 top-k logits。
BenchmarkResult RunBenchmark(const Model& model, ITokenizer& tokenizer,
                             const std::string& prompt, int n_predict,
                             const SamplingParams& params, bool verbose);

// 打印一次 benchmark 的耗时与吞吐。
void PrintBenchmark(const BenchmarkResult& result);

// 打印 logits 中最大的 k 个候选。
void PrintTopLogits(const Tensor& logits, int k);

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_BENCHMARK_H_
