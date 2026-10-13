#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_MODEL_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_MODEL_H_
#include <string>
#include <vector>
#include "mini_llama/tensor.h"
namespace mini_lla {
// Tiny 模型的结构参数，第一版只支持 CPU F32 权重。
struct ModelConfig {
  int vocab_size = 128;
  int dim = 32;
  int hidden_dim = 86;
  int n_layers = 2;
  int n_heads = 4;
  int n_kv_heads = 4;
  int max_seq_len = 128;
  float rope_theta = 10000.0f;
  float rms_norm_eps = 1e-5f;
  // 返回单个 attention head 的维度。
  int head_dim() const { return dim / n_heads; }
};
// 每层 Transformer 所需的线性层和归一化权重。
struct LayerWeights {
  Tensor attention_norm;
  Tensor wq;
  Tensor wk;
  Tensor wv;
  Tensor wo;
  Tensor ffn_norm;
  Tensor w_gate;
  Tensor w_up;
  Tensor w_down;
};
// 模型权重的内存表示，字段顺序与 model.bin 的导出顺序一致。
struct Model {
  ModelConfig config;
  Tensor token_embedding;
  Tensor final_norm;
  Tensor lm_head;
  std::vector<LayerWeights> layers;
};
// 读取 JSON manifest 和对应的 F32 二进制权重。
Model load_model(const std::string& manifest_path,
                 const std::string& weights_path);
}  // namespace mini_lla
#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_MODEL_H_
