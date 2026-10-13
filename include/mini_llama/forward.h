#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_FORWARD_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_FORWARD_H_
#include "mini_llama/kv_cache.h"
#include "mini_llama/model.h"
namespace mini_lla {

// 输入一个 token，返回下一个 token 的词表 logits。
Tensor forward_token(Context& context, const Model& model, int token);
}  // namespace mini_lla
#endif
