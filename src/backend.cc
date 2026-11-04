#include "mini_llama/backend.h"

#include <stdexcept>

#include "mini_llama/cuda_runtime.h"

namespace mini_lla {

bool ParseBackend(const std::string& text, BackendKind* kind) {
  if (kind == nullptr) return false;
  if (text == "cpu") {
    *kind = BackendKind::kCpu;
    return true;
  }
  if (text == "cuda") {
    *kind = BackendKind::kCuda;
    return true;
  }
  return false;
}

const char* BackendName(BackendKind kind) {
  return kind == BackendKind::kCuda ? "cuda" : "cpu";
}

void ValidateBackend(const BackendConfig& config) {
  if (config.kind == BackendKind::kCpu) {
    if (config.device_set) {
      throw std::invalid_argument("--device 只能和 --backend cuda 一起使用");
    }
    return;
  }
  if (!CudaRuntimeBuilt()) {
    throw std::runtime_error(
        "CUDA 未启用: 请使用 -DMINI_LLA_CUDA=ON 重新构建");
  }
  const int devices = CudaDeviceCount();
  if (config.device < 0 || config.device >= devices) {
    throw std::out_of_range("CUDA 设备编号超出范围,可见设备数: " +
                            std::to_string(devices));
  }
}

std::string BackendNote(const BackendConfig& config) {
  if (config.kind == BackendKind::kCpu) return "compute: cpu";
  return "compute: cuda device " + std::to_string(config.device);
}

}  // namespace mini_lla
