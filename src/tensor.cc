#include "mini_llama/tensor.h"

#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace mini_lla {
namespace {

// 计算形状的元素总数,要求每一维为正且乘法不溢出。
size_t ElementCount(const std::vector<int>& shape) {
  size_t count = 1;
  for (int dimension : shape) {
    if (dimension <= 0) {
      throw std::invalid_argument("tensor shape must be positive");
    }
    const size_t size = static_cast<size_t>(dimension);
    if (count > std::numeric_limits<size_t>::max() / size) {
      throw std::invalid_argument("tensor shape is too large");
    }
    count *= size;
  }
  return count;
}

// 检查某一维的下标是否在范围内。
void CheckIndex(int index, int dimension, const char* caller) {
  if (index < 0 || index >= dimension) {
    throw std::out_of_range(std::string(caller) + ": index out of range");
  }
}

// 把形状数组格式化为形如 "[2, 3]" 的文本。
std::string ShapeText(const std::vector<int>& shape) {
  std::ostringstream output;
  output << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) output << ", ";
    output << shape[i];
  }
  output << "]";
  return output.str();
}

}  // namespace

Tensor::Tensor(std::vector<int> input_shape, float fill)
    : shape(std::move(input_shape)), data(ElementCount(shape), fill) {}

size_t Tensor::Index(const std::vector<int>& indices) const {
  if (indices.size() != shape.size()) {
    throw std::out_of_range("tensor index rank mismatch");
  }
  size_t flat = 0;
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    CheckIndex(indices[axis], shape[axis], "Tensor::Index");
    flat = flat * static_cast<size_t>(shape[axis]) +
           static_cast<size_t>(indices[axis]);
  }
  return flat;
}

float* Tensor::RowPtr(int row) {
  if (shape.size() != 2) {
    throw std::runtime_error("Tensor::RowPtr expects a 2D tensor");
  }
  CheckIndex(row, shape[0], "Tensor::RowPtr");
  return data.data() + static_cast<size_t>(row) * shape[1];
}

const float* Tensor::RowPtr(int row) const {
  return const_cast<Tensor*>(this)->RowPtr(row);
}

Tensor Tensor::Reshape(std::vector<int> new_shape) const {
  if (ElementCount(new_shape) != data.size()) {
    throw std::invalid_argument("reshape changes element count");
  }
  Tensor result;
  result.shape = std::move(new_shape);
  result.data = data;
  return result;
}

void Tensor::RequireShape(const std::vector<int>& expected,
                          const char* caller) const {
  if (shape == expected) {
    return;
  }
  throw std::runtime_error(std::string(caller) + ": expected shape " +
                           ShapeText(expected) + ", got " + ShapeString());
}

std::string Tensor::ShapeString() const {
  return ShapeText(shape);
}

}  // namespace mini_lla
