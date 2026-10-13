#include <iostream>
#include <stdexcept>

#include "mini_llama/forward.h"
#include "mini_llama/ops.h"
#include "mini_llama/tokenizer.h"
namespace mini_lla {
// 执行一次完整的 prompt prefill 和 greedy token generation。
int run_generate(const std::string& directory, const std::string& prompt,
                 int count) {
  // 生成流程：加载模型 -> 编码 prompt -> prefill -> greedy decode。
  Model model = load_model(directory + "/model.json", directory + "/model.bin");
  AsciiTokenizer tokenizer;
  std::vector<int> tokens = tokenizer.encode(prompt);
  if (tokens.size() + static_cast<size_t>(count) >
      static_cast<size_t>(model.config.max_seq_len))
    throw std::runtime_error("prompt and generation exceed context length");
  Context context(model.config);
  Tensor logits;
  for (int token : tokens) logits = forward_token(context, model, token);
  std::cout << prompt;
  for (int i = 0; i < count; ++i) {
    const int token = argmax(logits);
    if (token == tokenizer.eos_id()) break;
    std::cout << tokenizer.decode_token(token);
    logits = forward_token(context, model, token);
  }
  std::cout << "\n";
  return 0;
}
}  // namespace mini_lla
int main(int argc, char** argv) {
  try {
    if (argc < 2 || std::string(argv[1]) != "generate") {
      std::cerr
          << "usage: mini-lla generate --model DIR --prompt TEXT --n COUNT\n";
      return 2;
    }
    std::string dir = "models/tiny", prompt = "hello";
    int count = 8;
    for (int i = 2; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--model" && i + 1 < argc)
        dir = argv[++i];
      else if (arg == "--prompt" && i + 1 < argc)
        prompt = argv[++i];
      else if (arg == "--n" && i + 1 < argc)
        count = std::stoi(argv[++i]);
    }
    return mini_lla::run_generate(dir, prompt, count);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}
