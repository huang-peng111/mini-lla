#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_CUDA_RUNTIME_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_CUDA_RUNTIME_H_

#include <cstddef>
#include <string>

namespace mini_lla {

// CUDA 设备的基本信息。
struct CudaDeviceInfo {
  int id = 0;
  std::string name;
  int compute_major = 0;
  int compute_minor = 0;
  size_t total_memory = 0;
  int driver_version = 0;
  int runtime_version = 0;
};

// 当前构建是否启用了 CUDA。
bool CudaRuntimeBuilt();

// 返回可见的 CUDA 设备数量;未启用 CUDA 时抛出异常。
int CudaDeviceCount();

// 查询指定设备的信息;设备不存在时抛出异常。
CudaDeviceInfo CudaGetDeviceInfo(int device);

// 切换当前线程使用的设备。
void CudaSetDevice(int device);

// 检查上一次 CUDA 调用的返回值,失败时抛出异常。
void CudaCheckLastError(const char* caller);

// 把设备信息格式化成一行文本。
std::string CudaFormatDeviceInfo(const CudaDeviceInfo& info);

// RAII 显存缓冲区:析构时自动释放,禁止拷贝、允许移动。
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  // 析构时释放显存。
  ~DeviceBuffer();

  // 移动构造与移动赋值只转移所有权。
  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  // 重新分配 bytes 字节显存,并释放旧内容。
  void Reset(size_t bytes);

  // 释放显存并清空大小。
  void Clear();

  // 从主机内存上传数据,要求 bytes 不超过缓冲区大小。
  void Upload(const void* source, size_t bytes);

  // 把显存内容下载到主机内存。
  void Download(void* destination, size_t bytes) const;

  // 把整块显存清零。
  void Zero();

  // 返回显存指针、字节数与是否为空。
  void* data() const { return data_; }
  size_t size() const { return size_; }
  bool empty() const { return data_ == nullptr; }

 private:
  void* data_ = nullptr;
  size_t size_ = 0;
};

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_CUDA_RUNTIME_H_
