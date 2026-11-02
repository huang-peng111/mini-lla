#include "mini_llama/loader.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mini_llama/gguf.h"
#include "mini_llama/gguf_loader.h"
#include "mini_llama/gguf_tokenizer.h"
#include "mini_llama/prompt_builder.h"
#include "mini_llama/quant.h"
#include "mini_llama/quantized_tensor.h"

namespace mini_lla {
namespace {

// manifest 中一个 tensor 的描述。
struct TensorEntry {
  std::string name;
  std::vector<int> shape;
  size_t offset = 0;
  size_t byte_size = 0;
};

// 统计各存储格式的线性层数量。
struct WeightCounts {
  int f32 = 0;
  int q8_0 = 0;
  int q4_0 = 0;
  int q4_1 = 0;

  // 按权重类型累加计数。
  void Add(const QuantizedTensor& weight) {
    switch (weight.type) {
      case QuantType::kF32:
        ++f32;
        return;
      case QuantType::kQ8_0:
        ++q8_0;
        return;
      case QuantType::kQ4_0:
        ++q4_0;
        return;
      case QuantType::kQ4_1:
        ++q4_1;
        return;
    }
  }
};

// 读取整个文本文件,失败时抛出异常。
std::string ReadTextFile(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("无法打开文件: " + path);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

// 从 manifest 中读取整数配置,字段缺失时使用 fallback。
int ConfigNumber(const std::string& text, const std::string& key,
                 int fallback) {
  const size_t start = text.find('"' + key + '"');
  if (start == std::string::npos) return fallback;
  const size_t colon = text.find(':', start);
  if (colon == std::string::npos) return fallback;
  return static_cast<int>(std::strtol(text.c_str() + colon + 1, nullptr, 10));
}

// 取出对象中某个字符串字段的值,找不到时返回空串。
std::string StringField(const std::string& object, const std::string& key) {
  const size_t start = object.find('"' + key + '"');
  if (start == std::string::npos) return "";
  const size_t colon = object.find(':', start);
  if (colon == std::string::npos) return "";
  const size_t quote = object.find('"', colon + 1);
  if (quote == std::string::npos) return "";
  const size_t end = object.find('"', quote + 1);
  if (end == std::string::npos) return "";
  return object.substr(quote + 1, end - quote - 1);
}

// 取出对象中某个数值字段的值,找不到时返回 0。
size_t NumberField(const std::string& object, const std::string& key) {
  const size_t start = object.find('"' + key + '"');
  if (start == std::string::npos) return 0;
  const size_t colon = object.find(':', start);
  if (colon == std::string::npos) return 0;
  return static_cast<size_t>(
      std::strtoull(object.c_str() + colon + 1, nullptr, 10));
}

// 取出对象中某个整数数组字段,找不到时返回空数组。
std::vector<int> NumberArrayField(const std::string& object,
                                  const std::string& key) {
  const size_t start = object.find('"' + key + '"');
  if (start == std::string::npos) return {};
  const size_t open = object.find('[', start);
  const size_t close = object.find(']', start);
  if (open == std::string::npos || close == std::string::npos) return {};

  std::vector<int> values;
  std::istringstream stream(object.substr(open + 1, close - open - 1));
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) values.push_back(std::atoi(item.c_str()));
  }
  return values;
}

// 解析 manifest 的 tensors 数组,每一项是一个 tensor 描述对象。
std::vector<TensorEntry> ParseTensorEntries(const std::string& text) {
  const size_t key = text.find("\"tensors\"");
  if (key == std::string::npos) {
    throw std::runtime_error("manifest 缺少 tensors 字段");
  }

  std::vector<TensorEntry> entries;
  size_t position = text.find('[', key);
  if (position == std::string::npos) {
    throw std::runtime_error("manifest 的 tensors 字段格式不正确");
  }
  while (true) {
    const size_t open = text.find('{', position);
    if (open == std::string::npos) break;
    const size_t close = text.find('}', open);
    if (close == std::string::npos) {
      throw std::runtime_error("manifest 的 tensor 条目未闭合");
    }

    const std::string object = text.substr(open, close - open + 1);
    TensorEntry entry;
    entry.name = StringField(object, "name");
    entry.shape = NumberArrayField(object, "shape");
    entry.offset = NumberField(object, "offset");
    entry.byte_size = NumberField(object, "byte_size");
    if (entry.name.empty() || entry.shape.empty()) {
      throw std::runtime_error("manifest 的 tensor 条目缺少 name 或 shape");
    }
    entries.push_back(std::move(entry));
    position = close + 1;
  }
  return entries;
}

// 把 tensors 数组整理成 名称 -> 描述 的查找表。
std::map<std::string, TensorEntry> IndexTensorEntries(
    const std::string& text) {
  std::map<std::string, TensorEntry> entries;
  for (TensorEntry entry : ParseTensorEntries(text)) {
    const std::string name = entry.name;
    entries[name] = std::move(entry);
  }
  return entries;
}

// 按 manifest 记录的偏移读取一个 F32 张量,并校验形状和字节数。
Tensor ReadTensor(std::ifstream& input,
                  const std::map<std::string, TensorEntry>& entries,
                  const std::string& name, const std::vector<int>& shape) {
  const auto found = entries.find(name);
  if (found == entries.end()) {
    throw std::runtime_error("manifest 缺少 tensor: " + name);
  }
  const TensorEntry& entry = found->second;
  if (entry.shape != shape) {
    throw std::runtime_error("manifest 中 " + name + " 的形状与配置不符");
  }

  Tensor tensor(shape);
  const size_t bytes = tensor.Size() * sizeof(float);
  if (entry.byte_size != bytes) {
    throw std::runtime_error("manifest 中 " + name + " 的字节数不符");
  }
  input.seekg(static_cast<std::streamoff>(entry.offset));
  input.read(reinterpret_cast<char*>(tensor.data.data()),
             static_cast<std::streamsize>(bytes));
  if (!input) {
    throw std::runtime_error("权重文件不完整: " + name);
  }
  return tensor;
}

// 读取 F32 权重并包装成量化张量,便于与 GGUF 路径共用推理代码。
QuantizedTensor ReadWeight(std::ifstream& input,
                           const std::map<std::string, TensorEntry>& entries,
                           const std::string& name,
                           const std::vector<int>& shape) {
  return MakeF32Weight(ReadTensor(input, entries, name, shape));
}

// 判断目录里是否存在 tiny 模型文件。
bool HasTinyModel(const std::string& directory) {
  return std::filesystem::exists(directory + "/model.json") &&
         std::filesystem::exists(directory + "/model.bin");
}

// 在目录中查找一个 GGUF 文件,找不到时返回空串。
std::string FindGguf(const std::string& directory) {
  for (const auto& item : std::filesystem::directory_iterator(directory)) {
    if (item.is_regular_file() && item.path().extension() == ".gguf") {
      return item.path().string();
    }
  }
  return "";
}

}  // namespace

Model LoadTinyModel(const std::string& directory) {
  const std::string manifest_path = directory + "/model.json";
  const std::string weights_path = directory + "/model.bin";
  const std::string text = ReadTextFile(manifest_path);

  Model model;
  ModelConfig& config = model.config;
  config.vocab_size = ConfigNumber(text, "vocab_size", config.vocab_size);
  config.dim = ConfigNumber(text, "dim", config.dim);
  config.hidden_dim = ConfigNumber(text, "hidden_dim", config.hidden_dim);
  config.n_layers = ConfigNumber(text, "n_layers", config.n_layers);
  config.n_heads = ConfigNumber(text, "n_heads", config.n_heads);
  config.n_kv_heads = ConfigNumber(text, "n_kv_heads", config.n_heads);
  config.max_seq_len = ConfigNumber(text, "max_seq_len", config.max_seq_len);
  ValidateConfig(config);

  const std::map<std::string, TensorEntry> entries = IndexTensorEntries(text);
  std::ifstream input(weights_path, std::ios::binary);
  if (!input) throw std::runtime_error("无法打开权重文件: " + weights_path);

  const std::vector<int> norm_shape{config.dim};
  model.token_embedding = ReadWeight(input, entries, "token_embedding",
                                     {config.vocab_size, config.dim});
  model.layers.resize(static_cast<size_t>(config.n_layers));
  for (int layer = 0; layer < config.n_layers; ++layer) {
    LayerWeights& weights = model.layers[static_cast<size_t>(layer)];
    const std::string prefix = "layers." + std::to_string(layer) + ".";
    weights.attention_norm =
        ReadTensor(input, entries, prefix + "attention_norm", norm_shape);
    weights.wq = ReadWeight(input, entries, prefix + "wq",
                            {config.attention_dim(), config.dim});
    weights.wk =
        ReadWeight(input, entries, prefix + "wk", {config.kv_dim(), config.dim});
    weights.wv =
        ReadWeight(input, entries, prefix + "wv", {config.kv_dim(), config.dim});
    weights.wo = ReadWeight(input, entries, prefix + "wo",
                            {config.dim, config.attention_dim()});
    weights.ffn_norm =
        ReadTensor(input, entries, prefix + "ffn_norm", norm_shape);
    weights.w_gate = ReadWeight(input, entries, prefix + "w_gate",
                                {config.hidden_dim, config.dim});
    weights.w_up = ReadWeight(input, entries, prefix + "w_up",
                              {config.hidden_dim, config.dim});
    weights.w_down = ReadWeight(input, entries, prefix + "w_down",
                                {config.dim, config.hidden_dim});
  }
  model.final_norm = ReadTensor(input, entries, "final_norm", norm_shape);
  model.lm_head = ReadWeight(input, entries, "lm_head",
                             {config.vocab_size, config.dim});
  return model;
}

LoadedModel LoadModelAndTokenizer(const std::string& path) {
  LoadedModel loaded;
  if (std::filesystem::is_directory(path)) {
    if (HasTinyModel(path)) {
      loaded.model = LoadTinyModel(path);
      loaded.tokenizer = CreateAsciiTokenizer();
      return loaded;
    }
    const std::string gguf_path = FindGguf(path);
    if (gguf_path.empty()) {
      throw std::runtime_error(
          "目录中既没有 model.json/model.bin,也没有 .gguf 文件: " + path);
    }
    loaded.model = LoadGgufModel(gguf_path);
    loaded.tokenizer = CreateGgufTokenizer(gguf_path);
    loaded.chat_template = ChatTemplateFromGguf(gguf_path);
  } else {
    loaded.model = LoadGgufModel(path);
    loaded.tokenizer = CreateGgufTokenizer(path);
    loaded.chat_template = ChatTemplateFromGguf(path);
  }

  if (!loaded.tokenizer) {
    throw std::runtime_error("模型缺少 tokenizer 信息: " + path);
  }
  return loaded;
}

std::string ChatTemplateFromGguf(const std::string& path) {
  GgufReader reader;
  if (!reader.Load(path)) return "";
  // 目前只内置了 Qwen2 的 <|im_start|> 模板,其它模板退回通用标签。
  if (reader.String("general.architecture", "") == "qwen2") {
    return kQwen2Template;
  }
  return "";
}

void InspectModel(const Model& model, const std::string& source) {
  const ModelConfig& config = model.config;
  std::cout << "model: " << source << "\n"
            << "  vocab_size:   " << config.vocab_size << "\n"
            << "  dim:          " << config.dim << "\n"
            << "  hidden_dim:   " << config.hidden_dim << "\n"
            << "  n_layers:     " << config.n_layers << "\n"
            << "  n_heads:      " << config.n_heads << "\n"
            << "  n_kv_heads:   " << config.n_kv_heads << "\n"
            << "  head_dim:     " << config.head_dim() << "\n"
            << "  max_seq_len:  " << config.max_seq_len << "\n"
            << "  rope_theta:   " << config.rope_theta << "\n"
            << "  rope_type:    "
            << (config.rope_type == RopeType::kNeoX ? "neox" : "normal") << "\n"
            << "  tied_lm_head: " << (model.tied_lm_head ? "true" : "false")
            << "\n"
            << "  weight bytes: " << model.WeightBytes() << "\n";

  WeightCounts counts;
  counts.Add(model.token_embedding);
  if (!model.tied_lm_head) counts.Add(model.lm_head);
  for (const LayerWeights& layer : model.layers) {
    counts.Add(layer.wq);
    counts.Add(layer.wk);
    counts.Add(layer.wv);
    counts.Add(layer.wo);
    counts.Add(layer.w_gate);
    counts.Add(layer.w_up);
    counts.Add(layer.w_down);
  }
  std::cout << "  linear weights: F32=" << counts.f32
            << " Q8_0=" << counts.q8_0 << " Q4_0=" << counts.q4_0
            << " Q4_1=" << counts.q4_1 << "\n";
}

}  // namespace mini_lla
