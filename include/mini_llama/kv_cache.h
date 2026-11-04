#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_KV_CACHE_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_KV_CACHE_H_

#include "mini_llama/cuda_attention.h"
#include "mini_llama/model.h"
#include "mini_llama/tensor.h"

namespace mini_lla {

// 保存每层、每个位置的 attention key/value,避免重复计算历史 token。
struct KvCache {
  Tensor keys;    // [层数, 最大长度, kv head 数, head_dim]
  Tensor values;  // 与 keys 同形状

  // 按模型配置分配缓存。
  KvCache(int layers, int max_seq_len, int kv_heads, int head_dim);

  // 写入一个位置的 key 和 value,形状为 [kv_heads, head_dim]。
  void Write(int layer, int position, const Tensor& key, const Tensor& value);

  // 返回指定 key head 在给定位置的首地址。
  const float* KeyPtr(int layer, int position, int head) const;

  // 返回指定 value head 在给定位置的首地址。
  const float* ValuePtr(int layer, int position, int head) const;
};

// 一次推理请求的状态:KV cache 与当前正在处理的位置。
struct Context {
  KvCache cache;
  // CUDA 后端使用设备上的 KV cache,CPU 后端保持为空。
  CudaKvCache cuda_cache;
  int position = 0;

  // 按模型配置创建缓存,context 长度取 config.max_seq_len。
  explicit Context(const ModelConfig& config);
};

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_KV_CACHE_H_
