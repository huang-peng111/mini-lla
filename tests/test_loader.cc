#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "mini_llama/loader.h"
#include "test_main.h"

using namespace mini_lla;

// 覆盖 tiny loader 的正常路径和损坏文件路径。
void TestLoader() {
  const std::string directory = test::TinyModelDir();
  const Model model = LoadTinyModel(directory);
  CHECK_EQUAL(model.config.vocab_size, 128);
  CHECK_EQUAL(model.layers.size(), static_cast<size_t>(model.config.n_layers));
  CHECK_TRUE(model.WeightBytes() > 0);
  CHECK_TRUE(model.token_embedding.IsF32());

  // 自动识别格式并创建 tokenizer。
  const LoadedModel loaded = LoadModelAndTokenizer(directory);
  CHECK_TRUE(loaded.tokenizer != nullptr);
  CHECK_EQUAL(loaded.tokenizer->vocab_size(), 128);
  CHECK_TRUE(loaded.tokenizer->Encode("hi")[1] == 'h');

  // 目录不存在、manifest 缺失都要报错。
  CHECK_THROWS(LoadTinyModel("no-such-directory"));
  CHECK_THROWS(LoadModelAndTokenizer("no-such-directory"));

  // 权重文件被截断时报错。
  const std::string temp_dir = test::TempPath("mini_lla_loader_case");
  std::filesystem::create_directories(temp_dir);
  std::filesystem::copy_file(directory + "/model.json",
                             temp_dir + "/model.json",
                             std::filesystem::copy_options::overwrite_existing);
  {
    std::ofstream output(temp_dir + "/model.bin", std::ios::binary);
    output << "too short";
  }
  CHECK_THROWS(LoadTinyModel(temp_dir));
  std::filesystem::remove_all(temp_dir);
}
