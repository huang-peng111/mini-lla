#include "mini_llama/quantized_tensor.h"

#include <stdexcept>
#include <utility>

namespace mini_lla {

const char* QuantTypeName(QuantType type) {
  switch (type) {
    case QuantType::kF32:
      return "F32";
    case QuantType::kQ8_0:
      return "Q8_0";
    case QuantType::kQ4_0:
      return "Q4_0";
    case QuantType::kQ4_1:
      return "Q4_1";
  }
  return "UNKNOWN";
}

QuantizedTensor MakeF32Weight(Tensor tensor) {
  if (tensor.shape.size() != 2) {
    throw std::invalid_argument("MakeF32Weight expects a 2D tensor");
  }
  QuantizedTensor weight;
  weight.type = QuantType::kF32;
  weight.rows = tensor.shape[0];
  weight.cols = tensor.shape[1];
  weight.f32 = std::move(tensor);
  return weight;
}

size_t QuantizedBytes(const QuantizedTensor& weight) {
  switch (weight.type) {
    case QuantType::kF32:
      return weight.f32.Size() * sizeof(float);
    case QuantType::kQ8_0:
      return weight.q8_0.size() * sizeof(BlockQ8_0);
    case QuantType::kQ4_0:
      return weight.q4_0.size() * sizeof(BlockQ4_0);
    case QuantType::kQ4_1:
      return weight.q4_1.size() * sizeof(BlockQ4_1);
  }
  return 0;
}

}  // namespace mini_lla
