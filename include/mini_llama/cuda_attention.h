#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_CUDA_ATTENTION_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_CUDA_ATTENTION_H_

#include "mini_llama/cuda_tensor.h"

namespace mini_lla {

// GPU 上的 KV cache,布局为 [层数, 最大长度, kv head 数, head_dim]。
class CudaKvCache {
 public:
  CudaKvCache() = default;

  // 按模型配置分配显存;重复调用会重新分配并清零。
  void Reset(int layers, int max_seq, int kv_heads, int head_dim);

  // 把当前 token 的 k/v 写入指定层和位置。
  void Write(int layer, int position, const CudaTensor& key,
             const CudaTensor& value);

  // 释放显存。
  void Clear();

  // 是否还没有分配显存。
  bool empty() const { return keys_.empty(); }

  // 返回某一层 key/value 的起始指针。
  const float* Keys(int layer) const;
  const float* Values(int layer) const;

  // 返回缓存尺寸参数。
  int layers() const { return layers_; }
  int max_seq() const { return max_seq_; }
  int kv_heads() const { return kv_heads_; }
  int head_dim() const { return head_dim_; }

 private:
  DeviceBuffer keys_;
  DeviceBuffer values_;
  int layers_ = 0;
  int max_seq_ = 0;
  int kv_heads_ = 0;
  int head_dim_ = 0;
};

// 在 GPU 上计算当前位置的因果注意力,输出形状为 [n_heads, head_dim]。
CudaTensor CudaAttention(const CudaTensor& q, const CudaKvCache& cache,
                         int layer, int position, int n_heads, int n_kv_heads,
                         int head_dim);

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_CUDA_ATTENTION_H_
