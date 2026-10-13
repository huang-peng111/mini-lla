#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_TENSOR_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_TENSOR_H_

#include <cstddef>
#include <string>
#include <vector>
namespace mini_lla {

// 轻量级 F32 张量：数据按行主序连续存放。
struct Tensor {
  std::vector<int> shape;
  std::vector<float> data;
  Tensor() = default;
  explicit Tensor(std::vector<int> s, float fill = 0.0f);
  size_t size() const {
    return data.size();
  }
  // 将多维下标转换成连续内存中的一维下标。
  size_t index(const std::vector<int>& indices) const;
  float& at(const std::vector<int>& indices) {
    return data.at(index(indices));
  }
  float at(const std::vector<int>& indices) const {
    return data.at(index(indices));
  }
  float& at1(int i) {
    return at({i});
  }
  float at1(int i) const {
    return at({i});
  }
  float& at2(int i, int j) {
    return at({i, j});
  }
  float at2(int i, int j) const {
    return at({i, j});
  }
  // 只改变形状，不复制数据；元素总数必须保持不变。
  Tensor reshape(std::vector<int> new_shape) const;
  std::string shape_string() const;
};
}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_TENSOR_H_
