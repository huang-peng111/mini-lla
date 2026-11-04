#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_CUDA_TENSOR_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_CUDA_TENSOR_H_

#include <cstddef>
#include <vector>

#include "mini_llama/cuda_runtime.h"
#include "mini_llama/tensor.h"

namespace mini_lla {

// 设备上的 F32 张量:形状加显存缓冲区。
class CudaTensor {
 public:
  CudaTensor() = default;

  // 按 shape 在显存中分配空间。
  explicit CudaTensor(std::vector<int> shape);

  // 从主机张量拷贝到显存。
  static CudaTensor FromHost(const Tensor& host);

  // 把显存内容拷贝回主机张量。
  Tensor Download() const;

  // 把整块显存清零。
  void Zero();

  // 返回形状与元素个数。
  const std::vector<int>& shape() const { return shape_; }
  size_t Size() const { return size_; }

  // 返回显存指针,便于传给内核。
  const float* data() const {
    return static_cast<const float*>(buffer_.data());
  }
  float* data() { return static_cast<float*>(buffer_.data()); }

 private:
  std::vector<int> shape_;
  size_t size_ = 0;
  DeviceBuffer buffer_;
};

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_CUDA_TENSOR_H_
