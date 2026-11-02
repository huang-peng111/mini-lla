#include "mini_llama/model.h"

#include <stdexcept>
#include <string>

namespace mini_lla {

const QuantizedTensor& Model::OutputWeight() const {
  return tied_lm_head ? token_embedding : lm_head;
}

size_t Model::WeightBytes() const {
  size_t bytes = QuantizedBytes(token_embedding) +
                 final_norm.Size() * sizeof(float);
  if (!tied_lm_head) {
    bytes += QuantizedBytes(lm_head);
  }
  for (const LayerWeights& layer : layers) {
    bytes += layer.attention_norm.Size() * sizeof(float);
    bytes += layer.ffn_norm.Size() * sizeof(float);
    bytes += QuantizedBytes(layer.wq) + QuantizedBytes(layer.wk) +
             QuantizedBytes(layer.wv) + QuantizedBytes(layer.wo) +
             QuantizedBytes(layer.w_gate) + QuantizedBytes(layer.w_up) +
             QuantizedBytes(layer.w_down);
    bytes += layer.bq.Size() * sizeof(float) + layer.bk.Size() * sizeof(float) +
             layer.bv.Size() * sizeof(float);
  }
  return bytes;
}

void ValidateConfig(const ModelConfig& config) {
  if (config.vocab_size <= 0 || config.dim <= 0 || config.hidden_dim <= 0 ||
      config.n_layers <= 0 || config.n_heads <= 0 || config.n_kv_heads <= 0 ||
      config.max_seq_len <= 0) {
    throw std::invalid_argument("model config must use positive dimensions");
  }
  if (config.dim % config.n_heads != 0) {
    throw std::invalid_argument("dim must be divisible by n_heads");
  }
  if (config.n_heads % config.n_kv_heads != 0) {
    throw std::invalid_argument("n_heads must be divisible by n_kv_heads");
  }
  if (config.head_dim() % 2 != 0) {
    throw std::invalid_argument("head_dim must be even for RoPE");
  }
}

}  // namespace mini_lla
