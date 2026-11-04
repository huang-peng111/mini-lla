#include "mini_llama/benchmark.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mini_llama/batch.h"
#include "mini_llama/forward.h"

namespace mini_lla {
namespace {

// 使用单调时钟计算从 start 到现在的毫秒数。
double ElapsedMs(std::chrono::steady_clock::time_point start) {
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

double BenchmarkResult::TokensPerSecond() const {
  const double total_ms = prefill_ms + decode_ms;
  if (total_ms <= 0.0) return 0.0;
  return generated_tokens * 1000.0 / total_ms;
}

double BenchmarkResult::DecodeTokensPerSecond() const {
  if (decode_ms <= 0.0 || decode_tokens <= 0) return 0.0;
  return decode_tokens * 1000.0 / decode_ms;
}

void PrintTopLogits(const Tensor& logits, int k) {
  if (logits.Rank() != 1) {
    throw std::invalid_argument("PrintTopLogits expects a vector");
  }
  k = std::max(1, std::min(k, static_cast<int>(logits.Size())));
  std::vector<std::pair<float, int>> candidates;
  candidates.reserve(logits.Size());
  for (size_t i = 0; i < logits.Size(); ++i) {
    candidates.emplace_back(logits.data[i], static_cast<int>(i));
  }
  std::partial_sort(candidates.begin(), candidates.begin() + k,
                    candidates.end(),
                    [](const auto& left, const auto& right) {
                      return left.first > right.first;
                    });
  std::cout << "  top-" << k << ":";
  for (int i = 0; i < k; ++i) {
    std::cout << " " << candidates[static_cast<size_t>(i)].second << "("
              << std::fixed << std::setprecision(3)
              << candidates[static_cast<size_t>(i)].first << ")";
  }
  std::cout << std::defaultfloat << "\n";
}

BenchmarkResult RunBenchmark(const Model& model, ITokenizer& tokenizer,
                             const std::string& prompt, int n_predict,
                             const SamplingParams& params, bool verbose) {
  if (prompt.empty()) throw std::invalid_argument("prompt is empty");
  const std::vector<int> prompt_tokens = tokenizer.Encode(prompt);
  if (prompt_tokens.empty()) throw std::invalid_argument("prompt is empty");
  if (prompt_tokens.size() >= static_cast<size_t>(model.config.max_seq_len)) {
    throw std::invalid_argument("prompt is longer than the context window");
  }

  BenchmarkResult result;
  result.prompt_tokens = static_cast<int>(prompt_tokens.size());
  Context context(model.config);
  Sampler sampler(params);

  // prefill:一次处理整个 prompt,得到第一个待采样位置的 logits。
  const auto prefill_start = std::chrono::steady_clock::now();
  Tensor logits = ForwardBatch(context, model,
                               MiniBatch::FromTokens(prompt_tokens, 0));
  result.prefill_ms = ElapsedMs(prefill_start);

  // decode:每次采样一个 token,再把它送回 forward。
  for (int i = 0; i < n_predict; ++i) {
    if (static_cast<int>(prompt_tokens.size()) + i >=
        model.config.max_seq_len) {
      break;
    }
    const int next = sampler.Sample(logits);
    ++result.generated_tokens;
    if (next == tokenizer.eos_id()) break;

    if (verbose) {
      std::cout << "  step " << i << " token=" << next << "\n";
      PrintTopLogits(logits, 5);
    }

    const auto decode_start = std::chrono::steady_clock::now();
    logits = ForwardToken(context, model, next);
    result.decode_ms += ElapsedMs(decode_start);
    ++result.decode_tokens;
  }
  return result;
}

void PrintBenchmark(const BenchmarkResult& result) {
  std::cout << std::fixed << std::setprecision(2)
            << "  prompt tokens:    " << result.prompt_tokens << "\n"
            << "  generated tokens: " << result.generated_tokens << "\n"
            << "  decode steps:     " << result.decode_tokens << "\n"
            << "  prefill time:     " << result.prefill_ms << " ms\n"
            << "  decode time:      " << result.decode_ms << " ms\n"
            << "  total time:       "
            << result.prefill_ms + result.decode_ms << " ms\n"
            << "  tokens/s (all):   " << result.TokensPerSecond() << "\n"
            << "  tokens/s (decode): " << result.DecodeTokensPerSecond()
            << std::defaultfloat << "\n";
}

}  // namespace mini_lla
