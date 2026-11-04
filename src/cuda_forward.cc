#include "mini_llama/cuda_forward.h"

#include <stdexcept>
#include <string>

#include "mini_llama/cuda_attention.h"
#include "mini_llama/cuda_matmul.h"
#include "mini_llama/cuda_ops.h"
#include "mini_llama/cuda_weights.h"

namespace mini_lla {
namespace {

// 取出已经上传到 GPU 的权重,缺失时抛出异常。
const CudaWeight& Weight(const Model& model, const std::string& name) {
  const CudaWeight* weight = model.cuda_weights->Find(name);
  if (weight == nullptr) {
    throw std::runtime_error("GPU 上缺少权重: " + name);
  }
  return *weight;
}

// 取出可选偏置,没有上传时返回 nullptr。
const CudaWeight* Bias(const Model& model, const std::string& name) {
  return model.cuda_weights->Find(name);
}

// 返回输出投影权重:共享权重时复用 token_embedding。
const CudaWeight& OutputWeight(const Model& model) {
  return Weight(model, model.tied_lm_head ? "token_embedding" : "lm_head");
}

}  // namespace

#ifdef MINI_LLA_USE_CUDA

Tensor ForwardTokenCuda(Context& context, const Model& model, int token) {
  if (model.cuda_weights == nullptr) {
    throw std::runtime_error("模型权重还没有上传到 GPU");
  }
  const ModelConfig& config = model.config;
  if (token < 0 || token >= config.vocab_size) {
    throw std::out_of_range("forward: token id out of range");
  }
  if (context.position < 0 || context.position >= config.max_seq_len) {
    throw std::out_of_range("forward: position exceeds context length");
  }
  if (context.cuda_cache.empty()) {
    context.cuda_cache.Reset(config.n_layers, config.max_seq_len,
                             config.n_kv_heads, config.head_dim());
  }

  CudaTensor x = CudaEmbeddingRow(Weight(model, "token_embedding"), token);
  for (int layer = 0; layer < config.n_layers; ++layer) {
    const std::string prefix = "layers." + std::to_string(layer) + ".";
    const CudaTensor norm =
        CudaRmsNorm(x, Weight(model, prefix + "attention_norm"),
                    config.rms_norm_eps);
    CudaTensor q = CudaAddBias(CudaLinear(norm, Weight(model, prefix + "wq")),
                               Bias(model, prefix + "bq"));
    CudaTensor k = CudaAddBias(CudaLinear(norm, Weight(model, prefix + "wk")),
                               Bias(model, prefix + "bk"));
    CudaTensor v = CudaAddBias(CudaLinear(norm, Weight(model, prefix + "wv")),
                               Bias(model, prefix + "bv"));
    CudaRope(&q, &k, context.position, config.rope_theta, config.n_heads,
             config.n_kv_heads, config.head_dim(), config.rope_type);

    context.cuda_cache.Write(layer, context.position, k, v);
    const CudaTensor attention =
        CudaAttention(q, context.cuda_cache, layer, context.position,
                      config.n_heads, config.n_kv_heads, config.head_dim());
    const CudaTensor x_attn =
        CudaAdd(x, CudaLinear(attention, Weight(model, prefix + "wo")));

    const CudaTensor ffn_norm =
        CudaRmsNorm(x_attn, Weight(model, prefix + "ffn_norm"),
                    config.rms_norm_eps);
    const CudaTensor gate = CudaLinear(ffn_norm, Weight(model, prefix + "w_gate"));
    const CudaTensor up = CudaLinear(ffn_norm, Weight(model, prefix + "w_up"));
    const CudaTensor ffn =
        CudaLinear(CudaMul(CudaSilu(gate), up),
                   Weight(model, prefix + "w_down"));
    x = CudaAdd(x_attn, ffn);
  }

  const CudaTensor logits =
      CudaLinear(CudaRmsNorm(x, Weight(model, "final_norm"),
                             config.rms_norm_eps),
                 OutputWeight(model));
  Tensor host = logits.Download();
  ++context.position;
  return host;
}

#else  // MINI_LLA_USE_CUDA

Tensor ForwardTokenCuda(Context&, const Model&, int) {
  throw std::runtime_error(
      "ForwardTokenCuda: CUDA 未启用,请使用 -DMINI_LLA_CUDA=ON 重新构建");
}

#endif  // MINI_LLA_USE_CUDA

}  // namespace mini_lla
