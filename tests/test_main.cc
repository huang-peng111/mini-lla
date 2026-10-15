#include <exception>
#include <iostream>
int test_tensor();
int test_ops();
int test_sampler_chat();
int test_gguf_quant();
// 运行当前阶段保留的最小测试集合。
int main() {
  try {
    if (test_tensor() != 0 || test_ops() != 0 || test_sampler_chat() != 0 ||
        test_gguf_quant() != 0)
      return 1;
  } catch (const std::exception& e) {
    std::cerr << "test failure: " << e.what() << "\n";
    return 1;
  }
  std::cout << "mini-lla tensor/op tests passed\n";
  return 0;
}
