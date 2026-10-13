#include "mini_llama/kv_cache.h"

#include <stdexcept>

#include "mini_llama/model.h"
namespace mini_lla {
// 按 [layer, position, head, dimension] 分配 key/value 缓存。
KvCache::KvCache(int layers, int sequence, int heads, int head_dim)
    : keys({layers, sequence, heads, head_dim}),
      values({layers, sequence, heads, head_dim}) {}
void KvCache::write(int layer, int position, const Tensor& key,
                    const Tensor& value) {
  // 当前 token 的 K/V 写入后，attention 才能读取到它。
  // 当前 token 的 K/V 写入后，attention 才能读取到它。
  if (key.shape != std::vector<int>{keys.shape[2], keys.shape[3]} ||
      value.shape != key.shape)
    throw std::invalid_argument("KV cache: shape mismatch");
  for (int h = 0; h < keys.shape[2]; ++h)
    for (int d = 0; d < keys.shape[3]; ++d) {
      keys.at({layer, position, h, d}) = key.at2(h, d);
      values.at({layer, position, h, d}) = value.at2(h, d);
    }
}
// 返回 key head 的连续内存地址。
const float* KvCache::key_ptr(int layer, int position, int head) const {
  const size_t index = keys.index({layer, position, head, 0});
  return keys.data.data() + index;
}
// 返回 value head 的连续内存地址。
const float* KvCache::value_ptr(int layer, int position, int head) const {
  const size_t index = values.index({layer, position, head, 0});
  return values.data.data() + index;
}
// 根据模型配置创建一次推理上下文。
Context::Context(const ModelConfig& config)
    : cache(config.n_layers, config.max_seq_len, config.n_kv_heads,
            config.head_dim()) {}
}  // namespace mini_lla
