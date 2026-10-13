#include "mini_llama/tensor.h"

#include <limits>
#include <sstream>
#include <stdexcept>
namespace mini_lla {
static size_t ElementCount(const std::vector<int>& shape) {
  size_t count = 1;
  for (int dimension : shape) {
    if (dimension <= 0 || count > std::numeric_limits<size_t>::max() /
                                      static_cast<size_t>(dimension)) {
      throw std::invalid_argument("invalid tensor shape");
    }
    count *= static_cast<size_t>(dimension);
  }
  return count;
}

// 按形状分配连续的 F32 存储。
Tensor::Tensor(std::vector<int> input_shape, float fill)
    : shape(std::move(input_shape)), data(ElementCount(shape), fill) {}

size_t Tensor::index(const std::vector<int>& indices) const {
  if (indices.size() != shape.size()) {
    throw std::out_of_range("tensor rank mismatch");
  }
  size_t flat_index = 0;
  for (size_t dimension = 0; dimension < shape.size(); ++dimension) {
    if (indices[dimension] < 0 || indices[dimension] >= shape[dimension]) {
      throw std::out_of_range("tensor index out of range");
    }
    flat_index = flat_index * static_cast<size_t>(shape[dimension]) +
                 static_cast<size_t>(indices[dimension]);
  }
  return flat_index;
}

// 创建同一数据数量的新形状张量。
Tensor Tensor::reshape(std::vector<int> new_shape) const {
  if (ElementCount(new_shape) != data.size()) {
    throw std::invalid_argument("reshape changes element count");
  }
  Tensor result;
  result.shape = std::move(new_shape);
  result.data = data;
  return result;
}

// 将形状格式化为日志可读的文本。
std::string Tensor::shape_string() const {
  std::ostringstream output;
  output << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) output << ", ";
    output << shape[i];
  }
  output << "]";
  return output.str();
}
}  // namespace mini_lla
