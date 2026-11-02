#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_BATCH_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_BATCH_H_

#include <vector>

namespace mini_lla {

// 一次前向的输入:token 及其位置。prompt 阶段多个 token,生成阶段一个。
struct MiniBatch {
  std::vector<int> tokens;
  std::vector<int> positions;

  // 返回 token 个数。
  int Size() const { return static_cast<int>(tokens.size()); }

  // 构造单 token 的生成输入。
  static MiniBatch Single(int token, int position);

  // 构造从 start 位置开始的多 token prompt 输入。
  static MiniBatch FromTokens(const std::vector<int>& tokens, int start);
};

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_BATCH_H_
