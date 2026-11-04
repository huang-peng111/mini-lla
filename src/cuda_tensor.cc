#include "mini_llama/cuda_tensor.h"

#include <stdexcept>
#include <utility>

namespace mini_lla {
namespace {

// 计算形状的元素总数。
size_t ElementCount(const std::vector<int>& shape) {
  size_t count = 1;
  for (int dimension : shape) {
    if (dimension <= 0) {
      throw std::invalid_argument("CudaTensor: shape must be positive");
    }
    count *= static_cast<size_t>(dimension);
  }
  return count;
}

}  // namespace

CudaTensor::CudaTensor(std::vector<int> shape) : shape_(std::move(shape)) {
  size_ = ElementCount(shape_);
  buffer_.Reset(size_ * sizeof(float));
}

CudaTensor CudaTensor::FromHost(const Tensor& host) {
  CudaTensor tensor(host.shape);
  if (host.Size() != tensor.Size()) {
    throw std::invalid_argument("CudaTensor::FromHost: size mismatch");
  }
  tensor.buffer_.Upload(host.data.data(), host.Size() * sizeof(float));
  return tensor;
}

Tensor CudaTensor::Download() const {
  Tensor host(shape_);
  buffer_.Download(host.data.data(), host.Size() * sizeof(float));
  return host;
}

void CudaTensor::Zero() { buffer_.Zero(); }

}  // namespace mini_lla
