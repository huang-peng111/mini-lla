#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_MODEL_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_MODEL_H_

#include <cstddef>
#include <vector>

#include "mini_llama/ops.h"
#include "mini_llama/quantized_tensor.h"
#include "mini_llama/tensor.h"

namespace mini_lla {

// 模型结构参数,来自 tiny manifest 或 GGUF metadata。
struct ModelConfig {
  int vocab_size = 128;
  int dim = 32;
  int hidden_dim = 64;
  int n_layers = 2;
  int n_heads = 4;
  int n_kv_heads = 4;
  int max_seq_len = 128;
  float rope_theta = 10000.0f;
  float rms_norm_eps = 1e-5f;
  RopeType rope_type = RopeType::kNormal;

  // 单个注意力头的维度。
  int head_dim() const { return dim / n_heads; }

  // 所有 query head 拼接后的维度。
  int attention_dim() const { return n_heads * head_dim(); }

  // 所有 key/value head 拼接后的维度,分组查询时小于 attention_dim。
  int kv_dim() const { return n_kv_heads * head_dim(); }
};

// 单层 Transformer 的权重;线性层可以是 F32 或量化存储。
struct LayerWeights {
  Tensor attention_norm;              // [dim]
  QuantizedTensor wq;                 // [attention_dim, dim]
  QuantizedTensor wk;                 // [kv_dim, dim]
  QuantizedTensor wv;                 // [kv_dim, dim]
  QuantizedTensor wo;                 // [dim, attention_dim]
  Tensor bq;                          // 可选偏置 [attention_dim]
  Tensor bk;                          // 可选偏置 [kv_dim]
  Tensor bv;                          // 可选偏置 [kv_dim]
  Tensor ffn_norm;                    // [dim]
  QuantizedTensor w_gate;             // [hidden_dim, dim]
  QuantizedTensor w_up;               // [hidden_dim, dim]
  QuantizedTensor w_down;             // [dim, hidden_dim]
};

// 完整模型权重。
struct Model {
  ModelConfig config;
  QuantizedTensor token_embedding;  // [vocab_size, dim]
  Tensor final_norm;                // [dim]
  QuantizedTensor lm_head;          // [vocab_size, dim]
  bool tied_lm_head = false;        // 为 true 时输出投影复用 token_embedding
  std::vector<LayerWeights> layers;

  // 返回实际使用的输出投影权重。
  const QuantizedTensor& OutputWeight() const;

  // 返回权重占用的字节数,用于 inspect 统计。
  size_t WeightBytes() const;
};

// 检查配置是否自洽,不合法时抛出 std::invalid_argument。
void ValidateConfig(const ModelConfig& config);

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_MODEL_H_
