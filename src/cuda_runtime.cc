#include "mini_llama/cuda_runtime.h"

#include <sstream>
#include <stdexcept>

#ifdef MINI_LLA_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace mini_lla {
namespace {

#ifndef MINI_LLA_USE_CUDA
// CUDA 未启用时统一的错误信息。
std::runtime_error CudaDisabled(const char* caller) {
  return std::runtime_error(std::string(caller) +
                            ": CUDA 未启用,请使用 -DMINI_LLA_CUDA=ON 重新构建");
}
#endif

}  // namespace

#ifdef MINI_LLA_USE_CUDA

void CudaCheckLastError(const char* caller) {
  const cudaError_t error = cudaGetLastError();
  if (error == cudaSuccess) return;
  throw std::runtime_error(std::string(caller) + ": " + cudaGetErrorString(error));
}

bool CudaRuntimeBuilt() { return true; }

int CudaDeviceCount() {
  int count = 0;
  const cudaError_t error = cudaGetDeviceCount(&count);
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("查询 CUDA 设备失败: ") +
                             cudaGetErrorString(error));
  }
  return count;
}

CudaDeviceInfo CudaGetDeviceInfo(int device) {
  cudaDeviceProp properties{};
  const cudaError_t error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("读取 CUDA 设备信息失败: ") +
                             cudaGetErrorString(error));
  }
  CudaDeviceInfo info;
  info.id = device;
  info.name = properties.name;
  info.compute_major = properties.major;
  info.compute_minor = properties.minor;
  info.total_memory = static_cast<size_t>(properties.totalGlobalMem);
  cudaDriverGetVersion(&info.driver_version);
  cudaRuntimeGetVersion(&info.runtime_version);
  return info;
}

void CudaSetDevice(int device) {
  const cudaError_t error = cudaSetDevice(device);
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("切换 CUDA 设备失败: ") +
                             cudaGetErrorString(error));
  }
}

DeviceBuffer::~DeviceBuffer() { Clear(); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : data_(other.data_), size_(other.size_) {
  other.data_ = nullptr;
  other.size_ = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    Clear();
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

void DeviceBuffer::Reset(size_t bytes) {
  Clear();
  if (bytes == 0) return;
  cudaError_t error = cudaMalloc(&data_, bytes);
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("分配显存失败: ") +
                             cudaGetErrorString(error));
  }
  size_ = bytes;
}

void DeviceBuffer::Clear() {
  if (data_ != nullptr) {
    cudaFree(data_);
    data_ = nullptr;
  }
  size_ = 0;
}

void DeviceBuffer::Upload(const void* source, size_t bytes) {
  if (bytes > size_ || source == nullptr) {
    throw std::invalid_argument("DeviceBuffer::Upload: size mismatch");
  }
  const cudaError_t error =
      cudaMemcpy(data_, source, bytes, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("上传显存失败: ") +
                             cudaGetErrorString(error));
  }
}

void DeviceBuffer::Download(void* destination, size_t bytes) const {
  if (bytes > size_ || destination == nullptr) {
    throw std::invalid_argument("DeviceBuffer::Download: size mismatch");
  }
  const cudaError_t error =
      cudaMemcpy(destination, data_, bytes, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("下载显存失败: ") +
                             cudaGetErrorString(error));
  }
}

void DeviceBuffer::Zero() {
  if (data_ == nullptr) return;
  cudaMemset(data_, 0, size_);
}

#else  // MINI_LLA_USE_CUDA

void CudaCheckLastError(const char* caller) { throw CudaDisabled(caller); }

bool CudaRuntimeBuilt() { return false; }

int CudaDeviceCount() { throw CudaDisabled("CudaDeviceCount"); }

CudaDeviceInfo CudaGetDeviceInfo(int) {
  throw CudaDisabled("CudaGetDeviceInfo");
}

void CudaSetDevice(int) { throw CudaDisabled("CudaSetDevice"); }

DeviceBuffer::~DeviceBuffer() = default;

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : data_(other.data_), size_(other.size_) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  data_ = other.data_;
  size_ = other.size_;
  return *this;
}

void DeviceBuffer::Reset(size_t) { throw CudaDisabled("DeviceBuffer"); }

void DeviceBuffer::Clear() {}

void DeviceBuffer::Upload(const void*, size_t) {
  throw CudaDisabled("DeviceBuffer::Upload");
}

void DeviceBuffer::Download(void*, size_t) const {
  throw CudaDisabled("DeviceBuffer::Download");
}

void DeviceBuffer::Zero() { throw CudaDisabled("DeviceBuffer::Zero"); }

#endif  // MINI_LLA_USE_CUDA

std::string CudaFormatDeviceInfo(const CudaDeviceInfo& info) {
  std::ostringstream text;
  text << "device " << info.id << ": " << info.name << ", compute "
       << info.compute_major << "." << info.compute_minor << ", "
       << (info.total_memory / (1024.0 * 1024.0)) << " MB, driver "
       << info.driver_version << ", runtime " << info.runtime_version;
  return text.str();
}

}  // namespace mini_lla
