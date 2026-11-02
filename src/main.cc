#include <algorithm>
#include <climits>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/batch.h"
#include "mini_llama/chat.h"
#include "mini_llama/forward.h"
#include "mini_llama/gguf.h"
#include "mini_llama/loader.h"
#include "mini_llama/prompt_builder.h"
#include "mini_llama/sampler.h"
#include "mini_llama/tokenizer.h"

namespace mini_lla {
namespace {

// 命令行选项,由 ParseOptions 填充。
struct Options {
  std::string command;
  std::string model = "models/tiny";
  std::string prompt = "hello";
  std::string chat_template;  // 空串表示按模型自动选择
  int n_predict = 8;
  int context_length = 0;  // 0 表示使用模型自带长度
  SamplingParams sampling;
};

// 打印总用法。
void PrintUsage(const char* program) {
  std::cout
      << "usage: " << program << " <command> [options]\n"
      << "commands:\n"
      << "  generate        one-shot text generation\n"
      << "  run             interactive multi-turn chat\n"
      << "  inspect         show model config and weight statistics\n"
      << "  inspect-gguf    show GGUF metadata and tensor table\n"
      << "usage examples:\n"
      << "  " << program << " generate --model models/tiny -p hello -n 8\n"
      << "  " << program << " run models/tiny --temperature 0.8 --top-k 8\n"
      << "  " << program << " inspect-gguf path/to/model.gguf\n"
      << "options:\n"
      << "  --model <path>       tiny directory or GGUF file "
         "(default: models/tiny)\n"
      << "  -p, --prompt <text>  prompt for generate (default: hello)\n"
      << "  -n, --n-predict <n>  tokens to generate (default: 8)\n"
      << "  --ctx <n>            override the context length\n"
      << "  --temperature <t>    sampling temperature, 0 means greedy\n"
      << "  --top-k <k>          top-k candidates, 0 means no limit\n"
      << "  --seed <s>           random seed, 0 means random\n"
      << "  --template <name>    chat template: qwen2 or plain\n";
}

// 取出选项后面紧跟的值,缺失时抛出异常。
std::string NextValue(int argc, char** argv, int* index,
                      const std::string& option) {
  if (*index + 1 >= argc) {
    throw std::invalid_argument("option " + option + " requires a value");
  }
  return argv[++(*index)];
}

// 解析整数并检查范围。
int ParseInt(const std::string& text, const char* name, int low, int high) {
  size_t parsed = 0;
  int value = 0;
  try {
    value = std::stoi(text, &parsed);
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string(name) +
                                " expects an integer, got: " + text);
  }
  if (parsed != text.size() || value < low || value > high) {
    throw std::invalid_argument(std::string(name) + " must be in [" +
                                std::to_string(low) + ", " +
                                std::to_string(high) + "], got: " + text);
  }
  return value;
}

// 解析浮点数并检查范围。
float ParseFloat(const std::string& text, const char* name, float low,
                 float high) {
  size_t parsed = 0;
  float value = 0.0f;
  try {
    value = std::stof(text, &parsed);
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string(name) +
                                " expects a number, got: " + text);
  }
  if (parsed != text.size() || value < low || value > high) {
    throw std::invalid_argument(std::string(name) + " is out of range: " + text);
  }
  return value;
}

// 解析命令行,遇到非法参数时抛出 std::invalid_argument。
Options ParseOptions(int argc, char** argv) {
  if (argc < 2) throw std::invalid_argument("missing command");

  Options options;
  options.command = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--model") {
      options.model = NextValue(argc, argv, &i, argument);
    } else if (argument == "-p" || argument == "--prompt") {
      options.prompt = NextValue(argc, argv, &i, argument);
    } else if (argument == "-n" || argument == "--n" ||
               argument == "--n-predict") {
      options.n_predict =
          ParseInt(NextValue(argc, argv, &i, argument), "--n-predict", 0, 4096);
    } else if (argument == "--ctx") {
      options.context_length =
          ParseInt(NextValue(argc, argv, &i, argument), "--ctx", 1, 65536);
    } else if (argument == "--temperature") {
      options.sampling.temperature =
          ParseFloat(NextValue(argc, argv, &i, argument), "--temperature", 0.0f,
                     1000.0f);
    } else if (argument == "--top-k") {
      options.sampling.top_k =
          ParseInt(NextValue(argc, argv, &i, argument), "--top-k", 0, 1000000);
    } else if (argument == "--seed") {
      options.sampling.seed = static_cast<unsigned int>(
          ParseInt(NextValue(argc, argv, &i, argument), "--seed", 0, INT_MAX));
    } else if (argument == "--template") {
      options.chat_template = NextValue(argc, argv, &i, argument);
    } else if (argument == "-h" || argument == "--help") {
      options.command = "help";
    } else if (argument[0] != '-') {
      // 允许把模型路径写成位置参数,例如 "inspect models/tiny"。
      options.model = argument;
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }
  if (options.chat_template == "plain") options.chat_template.clear();
  return options;
}

// 用命令行参数覆盖模型的上下文长度。
void ApplyContextLength(Model* model, int requested) {
  if (requested > 0) model->config.max_seq_len = requested;
}

// 逐 token 生成,解码成文本后流式打印。
// 返回生成的 token 数量。
int GenerateTokens(const Options& options, const Model& model,
                   ITokenizer& tokenizer, Context& context, Tensor logits,
                   int limit) {
  Sampler sampler(options.sampling);
  std::vector<int> generated;
  std::string printed;
  for (int i = 0; i < limit; ++i) {
    const int token = sampler.Sample(logits);
    if (token == tokenizer.eos_id()) break;
    generated.push_back(token);
    logits = ForwardToken(context, model, token);

    // 字节级 BPE 的一个 token 可能是半个字符,按累计文本的增量输出。
    const std::string text = tokenizer.Decode(generated);
    if (text.size() > printed.size()) {
      std::cout << text.substr(printed.size());
      std::cout.flush();
      printed = text;
    }
  }
  std::cout << "\n";
  return static_cast<int>(generated.size());
}

// generate 子命令:一次性完成 prompt 编码、prefill 和生成。
int RunGenerate(const Options& options) {
  LoadedModel loaded = LoadModelAndTokenizer(options.model);
  Model& model = loaded.model;
  ApplyContextLength(&model, options.context_length);
  ITokenizer& tokenizer = *loaded.tokenizer;

  const std::vector<int> tokens = tokenizer.Encode(options.prompt);
  if (tokens.empty()) throw std::runtime_error("prompt is empty");
  if (tokens.size() >= static_cast<size_t>(model.config.max_seq_len)) {
    throw std::runtime_error("prompt is longer than the context window");
  }
  const int room = model.config.max_seq_len - static_cast<int>(tokens.size());
  const int limit = std::min(options.n_predict, room);

  Context context(model.config);
  const Tensor logits = ForwardBatch(context, model,
                                     MiniBatch::FromTokens(tokens, 0));
  std::cout << "model: " << options.model << "\n"
            << "prompt tokens: " << tokens.size() << "\n"
            << "output: " << options.prompt;
  GenerateTokens(options, model, tokenizer, context, logits, limit);
  return 0;
}

// run 子命令:多轮聊天,每轮重新 prefill 完整历史。
int RunChat(const Options& options) {
  LoadedModel loaded = LoadModelAndTokenizer(options.model);
  Model& model = loaded.model;
  ApplyContextLength(&model, options.context_length);
  ITokenizer& tokenizer = *loaded.tokenizer;

  const std::string chat_template = options.chat_template.empty()
                                        ? loaded.chat_template
                                        : options.chat_template;
  ChatSession session;
  if (chat_template.empty()) {
    session.AddMessage("system", "You are a helpful assistant.");
  }

  std::cout << "mini-lla chat (model=" << options.model
            << ", template=" << (chat_template.empty() ? "plain" : chat_template)
            << ")\n"
            << "commands: /help /clear /exit\n";

  std::string input;
  while (true) {
    std::cout << "user> ";
    if (!std::getline(std::cin, input) || input == "/exit") break;
    if (input == "/help") {
      std::cout << "/help /clear /exit\n";
      continue;
    }
    if (input == "/clear") {
      session.Clear();
      if (chat_template.empty()) {
        session.AddMessage("system", "You are a helpful assistant.");
      }
      std::cout << "history cleared\n";
      continue;
    }
    if (input.empty()) continue;

    std::vector<ChatMessage> messages = session.Messages();
    messages.push_back({"user", input});
    const std::vector<int> tokens =
        tokenizer.Encode(BuildPrompt(messages, chat_template));
    if (tokens.size() >= static_cast<size_t>(model.config.max_seq_len)) {
      std::cout << "error: history is longer than the context window, "
                   "use /clear\n";
      continue;
    }
    const int room = model.config.max_seq_len - static_cast<int>(tokens.size());
    const int limit = std::min(options.n_predict, room);

    Context context(model.config);
    const Tensor logits = ForwardBatch(context, model,
                                       MiniBatch::FromTokens(tokens, 0));
    session.AddMessage("user", input);
    std::cout << "assistant> ";
    GenerateTokens(options, model, tokenizer, context, logits, limit);
  }
  return 0;
}

// inspect 子命令:打印模型配置和权重统计。
int RunInspect(const std::string& path) {
  LoadedModel loaded = LoadModelAndTokenizer(path);
  InspectModel(loaded.model, path);
  return 0;
}

// inspect-gguf 子命令:只读地打印 GGUF 文件内容。
int RunInspectGguf(const std::string& path) {
  GgufReader reader;
  if (!reader.Load(path)) {
    std::cerr << "failed to load GGUF: " << reader.Error() << "\n";
    return 1;
  }
  InspectGguf(path, reader);
  return 0;
}

}  // namespace
}  // namespace mini_lla

int main(int argc, char** argv) {
  using mini_lla::Options;
  try {
    const Options options = mini_lla::ParseOptions(argc, argv);
    if (options.command == "generate") return mini_lla::RunGenerate(options);
    if (options.command == "run") return mini_lla::RunChat(options);
    if (options.command == "inspect") return mini_lla::RunInspect(options.model);
    if (options.command == "inspect-gguf") {
      return mini_lla::RunInspectGguf(options.model);
    }
    if (options.command == "help" || options.command == "-h" ||
        options.command == "--help") {
      mini_lla::PrintUsage(argv[0]);
      return 0;
    }
    std::cerr << "unknown command: " << options.command << "\n";
    mini_lla::PrintUsage(argv[0]);
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}
