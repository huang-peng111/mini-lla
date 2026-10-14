#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/chat.h"
#include "mini_llama/forward.h"
#include "mini_llama/ops.h"
#include "mini_llama/sampler.h"
#include "mini_llama/tokenizer.h"

namespace mini_lla {

// 执行一次完整的 prompt prefill 和 greedy 或随机采样生成。
int run_generate(const std::string& directory, const std::string& prompt,
                 int count, const SamplingParams& sampling) {
  Model model = load_model(directory + "/model.json", directory + "/model.bin");
  AsciiTokenizer tokenizer;
  std::vector<int> tokens = tokenizer.encode(prompt);
  if (tokens.size() + static_cast<size_t>(count) >
      static_cast<size_t>(model.config.max_seq_len)) {
    throw std::runtime_error("prompt and generation exceed context length");
  }

  Context context(model.config);
  Sampler sampler(sampling);
  Tensor logits;
  for (int token : tokens) logits = forward_token(context, model, token);

  std::cout << prompt;
  for (int i = 0; i < count; ++i) {
    const int token = sampler.sample(logits);
    if (token == tokenizer.eos_id()) break;
    std::cout << tokenizer.decode_token(token);
    logits = forward_token(context, model, token);
  }
  std::cout << "\n";
  return 0;
}

// 运行一个简单的多轮终端聊天会话。
int run_chat(const std::string& directory, const SamplingParams& sampling) {
  Model model = load_model(directory + "/model.json", directory + "/model.bin");
  AsciiTokenizer tokenizer;
  ChatSession session;
  std::string input;

  while (true) {
    std::cout << "user> ";
    if (!std::getline(std::cin, input) || input == "/exit") break;
    if (input == "/clear") {
      session.clear();
      std::cout << "history cleared\n";
      continue;
    }
    if (input == "/help") {
      std::cout << "/help  /clear  /exit\n";
      continue;
    }

    session.add_message("user", input);
    const std::string prompt = session.build_prompt();
    const std::vector<int> tokens = tokenizer.encode(prompt);
    if (tokens.size() >= static_cast<size_t>(model.config.max_seq_len)) {
      std::cerr << "error: chat history is too long\n";
      continue;
    }

    Context context(model.config);
    Sampler sampler(sampling);
    Tensor logits;
    for (int token : tokens) logits = forward_token(context, model, token);

    std::string answer;
    for (int i = 0; i < 32; ++i) {
      const int token = sampler.sample(logits);
      if (token == tokenizer.eos_id()) break;
      answer += tokenizer.decode_token(token);
      logits = forward_token(context, model, token);
    }
    session.add_message("assistant", answer);
    std::cout << "assistant> " << answer << "\n";
  }
  return 0;
}

}  // namespace mini_lla

// 命令行入口：解析 generate/run 及采样相关参数。
int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::cerr << "usage: mini-lla generate|run [options]\n";
      return 2;
    }

    std::string directory = "models/tiny";
    std::string prompt = "hello";
    int count = 8;
    mini_lla::SamplingParams sampling;

    for (int i = 2; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--model" && i + 1 < argc) {
        directory = argv[++i];
      } else if (argument == "--prompt" && i + 1 < argc) {
        prompt = argv[++i];
      } else if (argument == "--n" && i + 1 < argc) {
        count = std::stoi(argv[++i]);
      } else if (argument == "--temperature" && i + 1 < argc) {
        sampling.temperature = std::stof(argv[++i]);
      } else if (argument == "--top-k" && i + 1 < argc) {
        sampling.top_k = std::stoi(argv[++i]);
      } else if (argument == "--seed" && i + 1 < argc) {
        sampling.seed = static_cast<unsigned int>(std::stoul(argv[++i]));
      }
    }

    if (std::string(argv[1]) == "generate") {
      return mini_lla::run_generate(directory, prompt, count, sampling);
    }
    if (std::string(argv[1]) == "run") {
      return mini_lla::run_chat(directory, sampling);
    }
    std::cerr << "unknown command: " << argv[1] << "\n";
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}
