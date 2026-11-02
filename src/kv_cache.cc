#include "mini_llama/kv_cache.h"

#include <stdexcept>

namespace mini_lla {
namespace {

// 检查 layer 与 position 是否落在缓存范围内。
void CheckRange(const Tensor& cache, int layer, int position) {
  if (layer < 0 || layer >= cache.shape[0]) {
    throw std::out_of_range("KvCache: layer out of range");
  }
  if (position < 0 || position >= cache.shape[1]) {
    throw std::out_of_range("KvCache: position out of range");
  }
}

}  // namespace

KvCache::KvCache(int layers, int max_seq_len, int kv_heads, int head_dim)
    : keys({layers, max_seq_len, kv_heads, head_dim}),
      values({layers, max_seq_len, kv_heads, head_dim}) {}

void KvCache::Write(int layer, int position, const Tensor& key,
                    const Tensor& value) {
  CheckRange(keys, layer, position);
  if (key.shape != value.shape || key.Rank() != 2 ||
      key.shape[0] != keys.shape[2] || key.shape[1] != keys.shape[3]) {
    throw std::invalid_argument("KvCache: key/value shape mismatch");
  }
  for (int head = 0; head < key.shape[0]; ++head) {
    for (int i = 0; i < key.shape[1]; ++i) {
      keys.At({layer, position, head, i}) = key.At2(head, i);
      values.At({layer, position, head, i}) = value.At2(head, i);
    }
  }
}

const float* KvCache::KeyPtr(int layer, int position, int head) const {
  CheckRange(keys, layer, position);
  if (head < 0 || head >= keys.shape[2]) {
    throw std::out_of_range("KvCache: head out of range");
  }
  return keys.data.data() + keys.Index({layer, position, head, 0});
}

const float* KvCache::ValuePtr(int layer, int position, int head) const {
  CheckRange(values, layer, position);
  if (head < 0 || head >= values.shape[2]) {
    throw std::out_of_range("KvCache: head out of range");
  }
  return values.data.data() + values.Index({layer, position, head, 0});
}

Context::Context(const ModelConfig& config)
    : cache(config.n_layers, config.max_seq_len, config.n_kv_heads,
            config.head_dim()) {}

}  // namespace mini_lla
