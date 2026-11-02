#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_FORWARD_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_FORWARD_H_

#include "mini_llama/batch.h"
#include "mini_llama/kv_cache.h"
#include "mini_llama/model.h"

namespace mini_lla {

// 处理一个 token,返回词表大小的 logits,并把 KV 写入缓存。
Tensor ForwardToken(Context& context, const Model& model, int token);

// 依次处理 batch 中的 token,返回最后一个 token 的 logits。
// prompt 阶段传入多个 token,生成阶段每次传入一个。
Tensor ForwardBatch(Context& context, const Model& model,
                    const MiniBatch& batch);

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_FORWARD_H_
