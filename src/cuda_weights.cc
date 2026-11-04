#include "mini_llama/cuda_weights.h"

#include <memory>
#include <stdexcept>
#include <utility>

#include "mini_llama/quant.h"

namespace mini_lla {

const CudaWeight* CudaModelWeights::Find(const std::string& name) const {
  for (const CudaWeight& weight : weights_) {
    if (weight.name == name) return &weight;
  }
  return nullptr;
}

void CudaModelWeights::Add(CudaWeight weight) {
  weights_.push_back(std::move(weight));
}

size_t CudaModelWeights::Bytes() const {
  size_t bytes = 0;
  for (const CudaWeight& weight : weights_) bytes += weight.buffer.size();
  return bytes;
}

#ifdef MINI_LLA_USE_CUDA

namespace {

// 把量化权重上传到显存。
void UploadQuantized(const std::string& name, const QuantizedTensor& weight,
                     CudaModelWeights* storage) {
  const size_t bytes = QuantizedBytes(weight);
  if (bytes == 0) {
    throw std::runtime_error("上传权重失败,数据为空: " + name);
  }

  CudaWeight uploaded;
  uploaded.name = name;
  uploaded.type = weight.type;
  uploaded.rows = weight.rows;
  uploaded.cols = weight.cols;
  uploaded.linear = true;
  uploaded.buffer.Reset(bytes);
  switch (weight.type) {
    case QuantType::kF32:
      uploaded.buffer.Upload(weight.f32.data.data(), bytes);
      break;
    case QuantType::kQ8_0:
      uploaded.buffer.Upload(weight.q8_0.data(), bytes);
      break;
    case QuantType::kQ4_0:
      uploaded.buffer.Upload(weight.q4_0.data(), bytes);
      break;
    case QuantType::kQ4_1:
      uploaded.buffer.Upload(weight.q4_1.data(), bytes);
      break;
  }
  storage->Add(std::move(uploaded));
}

// 把一维张量(norm 或 bias)上传到显存,空张量直接跳过。
void UploadVector(const std::string& name, const Tensor& tensor,
                  CudaModelWeights* storage) {
  if (tensor.Size() == 0) return;
  CudaWeight uploaded;
  uploaded.name = name;
  uploaded.type = QuantType::kF32;
  uploaded.rows = static_cast<int>(tensor.Size());
  uploaded.cols = 1;
  uploaded.linear = false;
  uploaded.buffer.Reset(tensor.Size() * sizeof(float));
  uploaded.buffer.Upload(tensor.data.data(), tensor.Size() * sizeof(float));
  storage->Add(std::move(uploaded));
}

}  // namespace

void UploadModelWeights(Model* model, int device) {
  if (model == nullptr) throw std::invalid_argument("模型指针为空");
  CudaSetDevice(device);

  auto storage = std::make_shared<CudaModelWeights>();
  storage->set_device(device);
  UploadQuantized("token_embedding", model->token_embedding, storage.get());
  UploadVector("final_norm", model->final_norm, storage.get());
  if (!model->tied_lm_head) {
    UploadQuantized("lm_head", model->lm_head, storage.get());
  }

  for (size_t layer = 0; layer < model->layers.size(); ++layer) {
    const LayerWeights& weights = model->layers[layer];
    const std::string prefix = "layers." + std::to_string(layer) + ".";
    UploadVector(prefix + "attention_norm", weights.attention_norm,
                 storage.get());
    UploadQuantized(prefix + "wq", weights.wq, storage.get());
    UploadQuantized(prefix + "wk", weights.wk, storage.get());
    UploadQuantized(prefix + "wv", weights.wv, storage.get());
    UploadQuantized(prefix + "wo", weights.wo, storage.get());
    UploadVector(prefix + "bq", weights.bq, storage.get());
    UploadVector(prefix + "bk", weights.bk, storage.get());
    UploadVector(prefix + "bv", weights.bv, storage.get());
    UploadVector(prefix + "ffn_norm", weights.ffn_norm, storage.get());
    UploadQuantized(prefix + "w_gate", weights.w_gate, storage.get());
    UploadQuantized(prefix + "w_up", weights.w_up, storage.get());
    UploadQuantized(prefix + "w_down", weights.w_down, storage.get());
  }
  model->cuda_weights = std::move(storage);
}

#else  // MINI_LLA_USE_CUDA

void UploadModelWeights(Model*, int) {
  throw std::runtime_error(
      "上传模型到 GPU 失败: CUDA 未启用,请使用 -DMINI_LLA_CUDA=ON 重新构建");
}

#endif  // MINI_LLA_USE_CUDA

void ClearModelWeights(Model* model) {
  if (model != nullptr) model->cuda_weights.reset();
}

size_t CudaWeightCount(const Model& model) {
  return model.cuda_weights ? model.cuda_weights->Count() : 0;
}

size_t CudaWeightBytes(const Model& model) {
  return model.cuda_weights ? model.cuda_weights->Bytes() : 0;
}

}  // namespace mini_lla
