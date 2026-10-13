#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_KV_CACHE_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_KV_CACHE_H_
#include "mini_llama/model.h"
#include "mini_llama/tensor.h"
namespace mini_lla {
// 保存每层每个位置的 attention key/value，避免重复计算历史 token。
struct KvCache {
  Tensor keys;
  Tensor values;
  // 按模型配置分配所有层的位置缓存。
  KvCache(int layers, int sequence, int heads, int head_dim);
  // 写入一个位置的 key/value。
  void write(int layer, int position, const Tensor& key, const Tensor& value);
  // 返回指定 key head 的连续内存地址。
  const float* key_ptr(int layer, int position, int head) const;
  // 返回指定 value head 的连续内存地址。
  const float* value_ptr(int layer, int position, int head) const;
};
// 一次生成请求的状态，包括 KV cache 和当前 token 位置。
struct Context {
  KvCache cache;
  int position = 0;
  // 创建一个与模型上下文长度匹配的推理状态。
  explicit Context(const ModelConfig& config);
};
}  // namespace mini_lla
#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_KV_CACHE_H_
