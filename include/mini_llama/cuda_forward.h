#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_CUDA_FORWARD_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_CUDA_FORWARD_H_

#include "mini_llama/kv_cache.h"
#include "mini_llama/model.h"

namespace mini_lla {

// 在 GPU 上执行一次单 token 前向,返回主机上的 logits。
// 调用前需要先通过 UploadModelWeights 上传权重。
Tensor ForwardTokenCuda(Context& context, const Model& model, int token);

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_CUDA_FORWARD_H_
