#include <algorithm>
#include <climits>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/backend.h"
#include "mini_llama/batch.h"
#include "mini_llama/benchmark.h"
#include "mini_llama/chat.h"
#include "mini_llama/cuda_runtime.h"
#include "mini_llama/cuda_weights.h"
#include "mini_llama/forward.h"
#include "mini_llama/gguf.h"
#include "mini_llama/loader.h"
#include "mini_llama/prompt_builder.h"
#include "mini_llama/sampler.h"
#include "mini_llama/thread_pool.h"
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
  int threads = 0;         // 0 表示使用硬件线程数
  bool verbose = false;
  BackendConfig backend;
  SamplingParams sampling;
};

// 打印总用法。
void PrintUsage(const char* program) {
  std::cout
      << "usage: " << program << " <command> [options]\n"
      << "commands:\n"
      << "  generate        one-shot text generation\n"
      << "  run             interactive multi-turn chat\n"
      << "  bench           measure prefill and decode speed\n"
      << "  inspect         show model config and weight statistics\n"
      << "  inspect-gguf    show GGUF metadata and tensor table\n"
      << "  --cuda-info     print CUDA device information\n"
      << "options:\n"
      << "  --model <path>       tiny directory or GGUF file "
         "(default: models/tiny)\n"
      << "  -p, --prompt <text>  prompt for generate/bench (default: hello)\n"
      << "  -n, --n-predict <n>  tokens to generate (default: 8)\n"
      << "  --ctx <n>            override the context length\n"
      << "  --threads <n>        CPU worker threads, 0 means auto\n"
      << "  --backend cpu|cuda   execution backend (default: cpu)\n"
      << "  --device <n>         CUDA device index (default: 0)\n"
      << "  --temperature <t>    sampling temperature, 0 means greedy\n"
      << "  --top-k <k>          top-k candidates, 0 means no limit\n"
      << "  --seed <s>           random seed, 0 means random\n"
      << "  --template <name>    chat template: qwen2 or plain\n"
      << "  --verbose            print per-step logits during bench\n";
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
    } else if (argument == "--threads") {
      options.threads =
          ParseInt(NextValue(argc, argv, &i, argument), "--threads", 0, 256);
    } else if (argument == "--backend") {
      const std::string value = NextValue(argc, argv, &i, argument);
      if (!ParseBackend(value, &options.backend.kind)) {
        throw std::invalid_argument("--backend must be cpu or cuda");
      }
    } else if (argument == "--device") {
      options.backend.device =
          ParseInt(NextValue(argc, argv, &i, argument), "--device", 0, 64);
      options.backend.device_set = true;
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
    } else if (argument == "--verbose") {
      options.verbose = true;
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

// 校验后端、设置线程数,并在需要时把权重上传到 GPU。
void PrepareModel(Model* model, const Options& options) {
  ValidateBackend(options.backend);
  SetThreadCount(options.threads);
  if (options.backend.kind == BackendKind::kCuda) {
    UploadModelWeights(model, options.backend.device);
  }
}

// 打印后端、线程数和权重概况。
void PrintModelSummary(const Model& model, const Options& options) {
  std::cout << "model: " << options.model << "\n"
            << "backend: " << BackendName(options.backend.kind) << "\n"
            << "threads: " << ThreadCount() << "\n"
            << "weight bytes: " << model.WeightBytes() << "\n";
  if (options.backend.kind == BackendKind::kCuda) {
    std::cout << "gpu weight bytes: " << CudaWeightBytes(model)
              << " (" << CudaWeightCount(model) << " tensors)\n";
  }
}

// 逐 token 生成,把解码结果流式打印;返回生成的 token 数。
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

// generate 子命令:编码 prompt、prefill,然后生成。
int RunGenerate(const Options& options) {
  LoadedModel loaded = LoadModelAndTokenizer(options.model);
  Model& model = loaded.model;
  ApplyContextLength(&model, options.context_length);
  PrepareModel(&model, options);
  ITokenizer& tokenizer = *loaded.tokenizer;

  const std::vector<int> tokens = tokenizer.Encode(options.prompt);
  if (tokens.empty()) throw std::runtime_error("prompt is empty");
  if (tokens.size() >= static_cast<size_t>(model.config.max_seq_len)) {
    throw std::runtime_error("prompt is longer than the context window");
  }
  const int room = model.config.max_seq_len - static_cast<int>(tokens.size());
  const int limit = std::min(options.n_predict, room);

  PrintModelSummary(model, options);
  Context context(model.config);
  const Tensor logits = ForwardBatch(context, model,
                                     MiniBatch::FromTokens(tokens, 0));
  std::cout << "prompt tokens: " << tokens.size() << "\n"
            << "output: " << options.prompt;
  GenerateTokens(options, model, tokenizer, context, logits, limit);
  return 0;
}

// run 子命令:多轮聊天,每轮重新 prefill 完整历史。
int RunChat(const Options& options) {
  LoadedModel loaded = LoadModelAndTokenizer(options.model);
  Model& model = loaded.model;
  ApplyContextLength(&model, options.context_length);
  PrepareModel(&model, options);
  ITokenizer& tokenizer = *loaded.tokenizer;

  const std::string chat_template = options.chat_template.empty()
                                        ? loaded.chat_template
                                        : options.chat_template;
  ChatSession session;
  if (chat_template.empty()) {
    session.AddMessage("system", "You are a helpful assistant.");
  }

  PrintModelSummary(model, options);
  std::cout << "template: " << (chat_template.empty() ? "plain" : chat_template)
            << "\ncommands: /help /clear /exit\n";

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

// bench 子命令:测量 prefill 与 decode 的耗时和吞吐。
int RunBench(const Options& options) {
  LoadedModel loaded = LoadModelAndTokenizer(options.model);
  Model& model = loaded.model;
  ApplyContextLength(&model, options.context_length);
  PrepareModel(&model, options);

  PrintModelSummary(model, options);
  std::cout << "benchmark:\n";
  const BenchmarkResult result =
      RunBenchmark(model, *loaded.tokenizer, options.prompt, options.n_predict,
                   options.sampling, options.verbose);
  PrintBenchmark(result);
  return 0;
}

// inspect 子命令:打印模型配置和权重统计。
int RunInspect(const Options& options) {
  LoadedModel loaded = LoadModelAndTokenizer(options.model);
  Model& model = loaded.model;
  ApplyContextLength(&model, options.context_length);
  PrepareModel(&model, options);
  InspectModel(model, options.model);
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

// 打印 CUDA 设备信息;CPU-only 构建给出明确提示。
int RunCudaInfo() {
  if (!CudaRuntimeBuilt()) {
    std::cout << "CUDA: 未启用(使用 -DMINI_LLA_CUDA=ON 构建后可用)\n";
    return 0;
  }
  const int count = CudaDeviceCount();
  std::cout << "CUDA devices: " << count << "\n";
  for (int device = 0; device < count; ++device) {
    std::cout << "  " << CudaFormatDeviceInfo(CudaGetDeviceInfo(device))
              << "\n";
  }
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
    if (options.command == "bench") return mini_lla::RunBench(options);
    if (options.command == "inspect") return mini_lla::RunInspect(options);
    if (options.command == "inspect-gguf") {
      return mini_lla::RunInspectGguf(options.model);
    }
    if (options.command == "--cuda-info") return mini_lla::RunCudaInfo();
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
