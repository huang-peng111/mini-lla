#include "mini_llama/forward.h"

#include <cmath>
#include <stdexcept>

#include "mini_llama/ops.h"
namespace mini_lla {
namespace {
// 根据 token id 从 embedding 矩阵取出隐藏向量。
Tensor embedding(const Model& m, int token) {
  // embedding 是词表矩阵按 token id 取出的一行。
  if (token < 0 || token >= m.config.vocab_size)
    throw std::out_of_range("token id out of range");
  Tensor x({m.config.dim});
  for (int i = 0; i < m.config.dim; ++i)
    x.data[i] = m.token_embedding.at2(token, i);
  return x;
}
// 将一维向量包装为一行矩阵后执行线性层。
Tensor vector_linear(const Tensor& x, const Tensor& weight) {
  Tensor row = x.reshape({1, x.shape[0]});
  return linear(row, weight).reshape({weight.shape[0]});
}
// 写入当前 K/V，并计算当前位置可见范围内的 causal attention。
Tensor attention(Context& ctx, const Model& m, const Tensor& q, const Tensor& k,
                 const Tensor& v, int layer) {
  // 只对当前位置及之前的位置计算分数，形成 causal attention。
  ctx.cache.write(layer, ctx.position, k, v);
  Tensor output({m.config.dim});
  const int heads = m.config.n_heads, hd = m.config.head_dim();
  for (int h = 0; h < heads; ++h) {
    std::vector<float> scores(ctx.position + 1);
    for (int t = 0; t <= ctx.position; ++t) {
      const float* kp = ctx.cache.key_ptr(layer, t, h);
      for (int d = 0; d < hd; ++d) scores[t] += q.at2(h, d) * kp[d];
      scores[t] /= std::sqrt(static_cast<float>(hd));
    }
    Tensor probs({ctx.position + 1});
    probs.data = scores;
    probs = softmax(probs);
    for (int d = 0; d < hd; ++d)
      for (int t = 0; t <= ctx.position; ++t)
        output.data[h * hd + d] +=
            probs.data[t] * ctx.cache.value_ptr(layer, t, h)[d];
  }
  return output;
}
}  // namespace
// 执行一个 token 的完整 Transformer 前向并返回 logits。
Tensor forward_token(Context& ctx, const Model& model, int token) {
  if (ctx.position >= model.config.max_seq_len)
    throw std::runtime_error("context length exceeded");
  // 依次执行每个 Transformer block，并在 block 内保存 KV cache。
  Tensor hidden = embedding(model, token);
  for (int layer = 0; layer < model.config.n_layers; ++layer) {
    const auto& w = model.layers[layer];
    Tensor norm = rms_norm(hidden, w.attention_norm, model.config.rms_norm_eps);
    Tensor q = vector_linear(norm, w.wq)
                   .reshape({model.config.n_heads, model.config.head_dim()});
    Tensor k = vector_linear(norm, w.wk)
                   .reshape({model.config.n_kv_heads, model.config.head_dim()});
    Tensor v = vector_linear(norm, w.wv)
                   .reshape({model.config.n_kv_heads, model.config.head_dim()});
    rope(q, k, ctx.position, model.config.rope_theta);
    Tensor attn = vector_linear(attention(ctx, model, q, k, v, layer), w.wo);
    hidden = add(hidden, attn);
    Tensor ff_norm = rms_norm(hidden, w.ffn_norm, model.config.rms_norm_eps);
    Tensor ff = vector_linear(swiglu(vector_linear(ff_norm, w.w_gate),
                                     vector_linear(ff_norm, w.w_up)),
                              w.w_down);
    hidden = add(hidden, ff);
  }
  Tensor logits = vector_linear(
      rms_norm(hidden, model.final_norm, model.config.rms_norm_eps),
      model.lm_head);
  ++ctx.position;
  return logits;
}
}  // namespace mini_lla
