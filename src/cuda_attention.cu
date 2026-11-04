#include "mini_llama/cuda_attention.h"

#include <cuda_runtime.h>

#include <stdexcept>

#include "mini_llama/cuda_runtime.h"

namespace mini_lla {
namespace {

// 把当前 token 的 key/value 写入缓存中的一个位置。
__global__ void KvWriteKernel(const float* key, const float* value, float* keys,
                              float* values, int layer, int position,
                              int kv_heads, int head_dim, int max_seq) {
  const int count = kv_heads * head_dim;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const size_t base =
      (static_cast<size_t>(layer) * max_seq + position) * count;
  keys[base + index] = key[index];
  values[base + index] = value[index];
}

// 一个 block 负责一个 query head:先算分数,再 softmax,最后加权求和。
__global__ void AttentionKernel(const float* q, const float* keys,
                                const float* values, float* output,
                                int position, int head_dim, int kv_heads,
                                int group) {
  const int head = blockIdx.x;
  const int kv_head = head / group;
  const int visible = position + 1;
  const int position_stride = kv_heads * head_dim;
  extern __shared__ float shared[];
  float* scores = shared;
  __shared__ float reduction[256];

  // 每个线程负责一部分通道,先做部分点积,再在 block 内规约。
  for (int t = 0; t < visible; ++t) {
    const float* key =
        keys + static_cast<size_t>(t) * position_stride + kv_head * head_dim;
    float dot = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
      dot += q[head * head_dim + i] * key[i];
    }
    reduction[threadIdx.x] = dot;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (threadIdx.x < stride) {
        reduction[threadIdx.x] += reduction[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      scores[t] = reduction[0] * rsqrtf(static_cast<float>(head_dim));
    }
    __syncthreads();
  }

  // 数值稳定的 softmax。
  float maximum = -INFINITY;
  for (int t = threadIdx.x; t < visible; t += blockDim.x) {
    maximum = fmaxf(maximum, scores[t]);
  }
  reduction[threadIdx.x] = maximum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] =
          fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  maximum = reduction[0];

  float total = 0.0f;
  for (int t = threadIdx.x; t < visible; t += blockDim.x) {
    scores[t] = expf(scores[t] - maximum);
    total += scores[t];
  }
  reduction[threadIdx.x] = total;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  total = reduction[0];
  for (int t = threadIdx.x; t < visible; t += blockDim.x) {
    scores[t] /= total;
  }
  __syncthreads();

  // 用注意力权重对 value 加权求和。
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    float sum = 0.0f;
    for (int t = 0; t < visible; ++t) {
      const float* value =
          values + static_cast<size_t>(t) * position_stride +
          kv_head * head_dim;
      sum += scores[t] * value[i];
    }
    output[head * head_dim + i] = sum;
  }
}

// 检查引用的 head 数量与缓存是否匹配。
void CheckCache(const CudaKvCache& cache, int layer, int position) {
  if (layer < 0 || layer >= cache.layers()) {
    throw std::out_of_range("CudaKvCache: layer out of range");
  }
  if (position < 0 || position >= cache.max_seq()) {
    throw std::out_of_range("CudaKvCache: position out of range");
  }
}

}  // namespace

void CudaKvCache::Reset(int layers, int max_seq, int kv_heads, int head_dim) {
  layers_ = layers;
  max_seq_ = max_seq;
  kv_heads_ = kv_heads;
  head_dim_ = head_dim;
  const size_t count = static_cast<size_t>(layers) * max_seq * kv_heads *
                       head_dim;
  keys_.Reset(count * sizeof(float));
  values_.Reset(count * sizeof(float));
  keys_.Zero();
  values_.Zero();
}

void CudaKvCache::Write(int layer, int position, const CudaTensor& key,
                        const CudaTensor& value) {
  CheckCache(*this, layer, position);
  const int count = kv_heads_ * head_dim_;
  if (key.Size() != static_cast<size_t>(count) ||
      value.Size() != static_cast<size_t>(count)) {
    throw std::invalid_argument("CudaKvCache::Write: tensor shape mismatch");
  }
  const int threads = 128;
  KvWriteKernel<<<(count + threads - 1) / threads, threads>>>(
      key.data(), value.data(), static_cast<float*>(keys_.data()),
      static_cast<float*>(values_.data()), layer, position, kv_heads_,
      head_dim_, max_seq_);
  CudaCheckLastError("CudaKvCache::Write");
}

void CudaKvCache::Clear() {
  keys_.Clear();
  values_.Clear();
  layers_ = max_seq_ = kv_heads_ = head_dim_ = 0;
}

const float* CudaKvCache::Keys(int layer) const {
  if (layer < 0 || layer >= layers_) {
    throw std::out_of_range("CudaKvCache::Keys: layer out of range");
  }
  const size_t offset = static_cast<size_t>(layer) * max_seq_ * kv_heads_ *
                        head_dim_;
  return static_cast<const float*>(keys_.data()) + offset;
}

const float* CudaKvCache::Values(int layer) const {
  if (layer < 0 || layer >= layers_) {
    throw std::out_of_range("CudaKvCache::Values: layer out of range");
  }
  const size_t offset = static_cast<size_t>(layer) * max_seq_ * kv_heads_ *
                        head_dim_;
  return static_cast<const float*>(values_.data()) + offset;
}

CudaTensor CudaAttention(const CudaTensor& q, const CudaKvCache& cache,
                         int layer, int position, int n_heads, int n_kv_heads,
                         int head_dim) {
  CheckCache(cache, layer, position);
  if (cache.kv_heads() != n_kv_heads || cache.head_dim() != head_dim) {
    throw std::invalid_argument("CudaAttention: cache shape mismatch");
  }
  if (q.Size() != static_cast<size_t>(n_heads * head_dim)) {
    throw std::invalid_argument("CudaAttention: query shape mismatch");
  }

  const int group = n_heads / n_kv_heads;
  const int visible = position + 1;
  const size_t shared_bytes = static_cast<size_t>(visible) * sizeof(float);
  CudaTensor output({n_heads, head_dim});
  AttentionKernel<<<n_heads, 256, shared_bytes>>>(
      q.data(), cache.Keys(layer), cache.Values(layer), output.data(), position,
      head_dim, cache.kv_heads(), group);
  CudaCheckLastError("CudaAttention");
  return output;
}

}  // namespace mini_lla
