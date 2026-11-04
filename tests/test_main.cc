#include <exception>
#include <iostream>

// 各测试文件导出的入口。
void TestTensor();
void TestOps();
void TestKvCache();
void TestForward();
void TestLoader();
void TestTokenizer();
void TestSamplerChat();
void TestQuant();
void TestGguf();
void TestGgufLoader();
void TestThreadPool();
void TestBenchmark();
void TestCudaRuntime();

// 依次运行全部测试,统计失败数量并返回对应状态码。
int main() {
  const struct {
    const char* name;
    void (*run)();
  } tests[] = {
      {"tensor", TestTensor},           {"ops", TestOps},
      {"kv_cache", TestKvCache},        {"forward", TestForward},
      {"loader", TestLoader},           {"tokenizer", TestTokenizer},
      {"sampler_chat", TestSamplerChat}, {"quant", TestQuant},
      {"gguf", TestGguf},               {"gguf_loader", TestGgufLoader},
      {"thread_pool", TestThreadPool},  {"benchmark", TestBenchmark},
      {"cuda_runtime", TestCudaRuntime},
  };

  int failed = 0;
  for (const auto& test : tests) {
    try {
      test.run();
      std::cout << "[ ok ] " << test.name << "\n";
    } catch (const std::exception& error) {
      ++failed;
      std::cout << "[fail] " << test.name << ": " << error.what() << "\n";
    }
  }
  std::cout << (failed == 0 ? "all tests passed\n" : "some tests failed\n");
  return failed == 0 ? 0 : 1;
}
