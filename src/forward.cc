#include "mini_llama/forward.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "mini_llama/cuda_forward.h"
#include "mini_llama/ops.h"
#include "mini_llama/quant.h"
#include "mini_llama/thread_pool.h"

namespace mini_lla {
namespace {

// 检查 token、位置和层数是否与模型配置一致。
void ValidateInputs(const Context& context, const Model& model, int token) {
  if (token < 0 || token >= model.config.vocab_size) {
    throw std::out_of_range("forward: token id out of range");
  }
  if (context.position < 0 || context.position >= model.config.max_seq_len) {
    throw std::out_of_range("forward: position exceeds context length");
  }
  if (model.layers.size() != static_cast<size_t>(model.config.n_layers)) {
    throw std::runtime_error("forward: layer count does not match config");
  }
}

// 取出 token 对应的嵌入向量,量化权重时只反量化这一行。
Tensor Embedding(const Model& model, int token) {
  Tensor vector({model.config.dim});
  DequantizeRow(model.token_embedding, token, vector.data.data());
  return vector;
}

// 加上可选偏置:偏置为空时直接返回原值。
Tensor AddBias(Tensor value, const Tensor& bias) {
  if (bias.Size() == 0) {
    return value;
  }
  if (bias.Size() != value.Size()) {
    throw std::invalid_argument("forward: bias shape mismatch");
  }
  for (size_t i = 0; i < value.Size(); ++i) {
    value.data[i] += bias.data[i];
  }
  return value;
}

// 计算一个 query head 在当前可见范围内的注意力结果。
void AttentionHead(const Context& context, const Tensor& q, int layer,
                   int head, int kv_head, int head_dim, int visible,
                   float scale, float* output) {
  Tensor scores({visible});
  for (int t = 0; t < visible; ++t) {
    const float* key = context.cache.KeyPtr(layer, t, kv_head);
    float dot = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
      dot += q.At2(head, i) * key[i];
    }
    scores.data[static_cast<size_t>(t)] = dot * scale;
  }

  const Tensor weights = Softmax(scores);
  for (int t = 0; t < visible; ++t) {
    const float* value = context.cache.ValuePtr(layer, t, kv_head);
    const float weight = weights.data[static_cast<size_t>(t)];
    for (int i = 0; i < head_dim; ++i) {
      output[i] += weight * value[i];
    }
  }
}

// 计算当前位置的因果注意力。
// 分组查询(GQA)中相邻 group 个 query head 共用一个 key/value head。
Tensor Attention(Context& context, const Model& model,
                 const Tensor& q, const Tensor& k, const Tensor& v,
                 int index) {
  const ModelConfig& config = model.config;
  context.cache.Write(index, context.position, k, v);

  const int head_dim = config.head_dim();
  const int group = config.n_heads / config.n_kv_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const int visible = context.position + 1;

  Tensor output({config.attention_dim()});
  // 每个 query head 独立计算,可以并行;写回的位置互不重叠。
  ParallelFor(config.n_heads, [&](int begin, int end) {
    for (int head = begin; head < end; ++head) {
      AttentionHead(context, q, index, head, head / group, head_dim, visible,
                    scale, output.data.data() + head * head_dim);
    }
  });
  return output;
}

// 计算 SwiGLU 前馈网络:SwiGlu(w_gate * h, w_up * h) * w_down。
Tensor FeedForward(const LayerWeights& layer, const Tensor& hidden) {
  const Tensor gate = Linear(hidden, layer.w_gate);
  const Tensor up = Linear(hidden, layer.w_up);
  return Linear(SwiGlu(gate, up), layer.w_down);
}

// 完成一层的注意力子层和前馈子层,含两条残差连接。
Tensor ForwardLayer(Context& context, const Model& model,
                    const LayerWeights& layer, const Tensor& hidden,
                    int index) {
  const ModelConfig& config = model.config;
  const int head_dim = config.head_dim();

  const Tensor norm = RmsNorm(hidden, layer.attention_norm,
                              config.rms_norm_eps);
  Tensor q = AddBias(Linear(norm, layer.wq), layer.bq)
                 .Reshape({config.n_heads, head_dim});
  Tensor k = AddBias(Linear(norm, layer.wk), layer.bk)
                 .Reshape({config.n_kv_heads, head_dim});
  Tensor v = AddBias(Linear(norm, layer.wv), layer.bv)
                 .Reshape({config.n_kv_heads, head_dim});
  Rope(q, k, context.position, config.rope_theta, config.rope_type);

  const Tensor attention = Attention(context, model, q, k, v, index);
  const Tensor result = Add(hidden, Linear(attention, layer.wo));

  const Tensor ffn_norm = RmsNorm(result, layer.ffn_norm, config.rms_norm_eps);
  return Add(result, FeedForward(layer, ffn_norm));
}

// 最终归一化后做输出投影,得到词表 logits。
Tensor ComputeLogits(const Model& model, const Tensor& hidden) {
  const Tensor norm =
      RmsNorm(hidden, model.final_norm, model.config.rms_norm_eps);
  return Linear(norm, model.OutputWeight());
}

}  // namespace

Tensor ForwardToken(Context& context, const Model& model, int token) {
  ValidateInputs(context, model, token);
  if (model.cuda_weights != nullptr) {
    return ForwardTokenCuda(context, model, token);
  }

  Tensor hidden = Embedding(model, token);
  for (int layer = 0; layer < model.config.n_layers; ++layer) {
    hidden = ForwardLayer(context, model, model.layers[layer], hidden, layer);
  }

  const Tensor logits = ComputeLogits(model, hidden);
  ++context.position;
  return logits;
}

}  // namespace mini_lla
