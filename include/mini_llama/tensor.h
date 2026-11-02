#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_TENSOR_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_TENSOR_H_

#include <cstddef>
#include <string>
#include <vector>

namespace mini_lla {

// 连续内存的 F32 张量:shape 描述形状,data 按行主序存放元素。
struct Tensor {
  std::vector<int> shape;
  std::vector<float> data;

  // 默认构造一个空张量。
  Tensor() = default;

  // 按 shape 分配元素,并用 fill 填充;形状非法时抛出异常。
  explicit Tensor(std::vector<int> shape, float fill = 0.0f);

  // 返回元素总数。
  size_t Size() const { return data.size(); }

  // 返回维度个数。
  int Rank() const { return static_cast<int>(shape.size()); }

  // 把多维下标转换为连续内存下标,下标越界时抛出异常。
  size_t Index(const std::vector<int>& indices) const;

  // 按多维下标读写元素。
  float& At(const std::vector<int>& indices) {
    return data[Index(indices)];
  }
  float At(const std::vector<int>& indices) const {
    return data[Index(indices)];
  }

  // 一维下标的读写快捷方式。
  float& At1(int i) { return At({i}); }
  float At1(int i) const { return At({i}); }

  // 二维下标的读写快捷方式。
  float& At2(int i, int j) { return At({i, j}); }
  float At2(int i, int j) const { return At({i, j}); }

  // 返回二维张量第 row 行的首地址。
  float* RowPtr(int row);
  const float* RowPtr(int row) const;

  // 保持元素总数不变地改变形状,返回新的张量(会复制数据)。
  Tensor Reshape(std::vector<int> new_shape) const;

  // 要求当前形状等于 expected,否则抛出异常;caller 用于错误信息。
  void RequireShape(const std::vector<int>& expected,
                    const char* caller) const;

  // 返回形如 "[2, 3]" 的形状文本。
  std::string ShapeString() const;
};

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_TENSOR_H_
