#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_CUDA_WEIGHTS_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_CUDA_WEIGHTS_H_

#include <cstddef>
#include <string>
#include <vector>

#include "mini_llama/cuda_runtime.h"
#include "mini_llama/model.h"
#include "mini_llama/quantized_tensor.h"

namespace mini_lla {

// GPU 上的一份权重:线性层权重或一维的 norm/bias。
struct CudaWeight {
  std::string name;
  QuantType type = QuantType::kF32;
  int rows = 0;
  int cols = 0;
  bool linear = true;  // false 表示 norm 或 bias
  DeviceBuffer buffer;
};

// 模型的 GPU 权重集合,按名称查找。
class CudaModelWeights {
 public:
  // 按名称查找权重,找不到返回 nullptr。
  const CudaWeight* Find(const std::string& name) const;

  // 加入一项权重。
  void Add(CudaWeight weight);

  // 返回权重数量与显存字节数。
  size_t Count() const { return weights_.size(); }
  size_t Bytes() const;

  // 读写所属设备编号。
  int device() const { return device_; }
  void set_device(int device) { device_ = device; }

 private:
  int device_ = 0;
  std::vector<CudaWeight> weights_;
};

// 把模型权重上传到 GPU,失败时抛出异常。
void UploadModelWeights(Model* model, int device);

// 释放模型的 GPU 权重。
void ClearModelWeights(Model* model);

// 返回已上传的权重数量与字节数,未上传时返回 0。
size_t CudaWeightCount(const Model& model);
size_t CudaWeightBytes(const Model& model);

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_CUDA_WEIGHTS_H_
