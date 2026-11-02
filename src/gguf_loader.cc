#include "mini_llama/gguf_loader.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mini_llama/gguf.h"
#include "mini_llama/quant.h"
#include "mini_llama/quantized_tensor.h"

namespace mini_lla {
namespace {

// GGUF 形状中第一个维度变化最快,二维权重需要交换两维得到 [输出, 输入]。
std::vector<int> MapShape(const std::vector<uint64_t>& shape) {
  std::vector<int> mapped;
  mapped.reserve(shape.size());
  if (shape.size() == 2) {
    mapped.push_back(static_cast<int>(shape[1]));
    mapped.push_back(static_cast<int>(shape[0]));
    return mapped;
  }
  for (uint64_t dimension : shape) {
    mapped.push_back(static_cast<int>(dimension));
  }
  return mapped;
}

// 读取 tensor 数据到目标缓冲区,失败时抛出异常。
void ReadTensorData(const std::string& path, const GgufReader& reader,
                    const GgufTensorInfo& info, void* destination,
                    size_t bytes) {
  if (!ReadGgufTensorInto(path, reader, info, destination, bytes)) {
    throw std::runtime_error("读取 GGUF tensor 失败: " + info.name);
  }
}

// 按给定的逻辑形状读取 tensor:量化类型保留 block,F32 单独处理。
QuantizedTensor LoadBlocks(const std::string& path, const GgufReader& reader,
                           const GgufTensorInfo& info, int rows, int cols) {
  uint64_t bytes = 0;
  if (!GgufTensorDataSize(info, &bytes)) {
    throw std::runtime_error("不支持的 GGUF tensor 类型: " + info.name + " (" +
                             GgmlTypeName(info.type) + ")");
  }

  QuantizedTensor weight;
  weight.rows = rows;
  weight.cols = cols;
  switch (info.type) {
    case GgmlType::kQ8_0:
      weight.type = QuantType::kQ8_0;
      weight.q8_0.resize(static_cast<size_t>(bytes) / sizeof(BlockQ8_0));
      ReadTensorData(path, reader, info, weight.q8_0.data(), bytes);
      return weight;
    case GgmlType::kQ4_0:
      weight.type = QuantType::kQ4_0;
      weight.q4_0.resize(static_cast<size_t>(bytes) / sizeof(BlockQ4_0));
      ReadTensorData(path, reader, info, weight.q4_0.data(), bytes);
      return weight;
    case GgmlType::kQ4_1:
      weight.type = QuantType::kQ4_1;
      weight.q4_1.resize(static_cast<size_t>(bytes) / sizeof(BlockQ4_1));
      ReadTensorData(path, reader, info, weight.q4_1.data(), bytes);
      return weight;
    case GgmlType::kF32:
    case GgmlType::kF16:
      break;
  }
  throw std::runtime_error("该 tensor 不是量化类型: " + info.name);
}

// 把 tensor 读成线性层权重:F32 或任意一种量化存储都支持。
QuantizedTensor LoadWeight(const std::string& path, const GgufReader& reader,
                           const GgufTensorInfo& info) {
  const std::vector<int> shape = MapShape(info.shape);
  if (shape.size() != 2) {
    throw std::runtime_error("GGUF 线性层权重必须是二维: " + info.name);
  }
  if (info.type == GgmlType::kF32) {
    Tensor tensor(shape);
    ReadTensorData(path, reader, info, tensor.data.data(),
                   tensor.Size() * sizeof(float));
    return MakeF32Weight(std::move(tensor));
  }
  return LoadBlocks(path, reader, info, shape[0], shape[1]);
}

// 把 tensor 读成 F32 张量,norm 和 bias 需要完整的 F32 数据。
Tensor LoadF32(const std::string& path, const GgufReader& reader,
               const GgufTensorInfo& info) {
  const std::vector<int> shape = MapShape(info.shape);
  if (info.type == GgmlType::kF32) {
    Tensor tensor(shape);
    ReadTensorData(path, reader, info, tensor.data.data(),
                   tensor.Size() * sizeof(float));
    return tensor;
  }
  if (info.type == GgmlType::kF16) {
    throw std::runtime_error("暂不支持 F16 tensor: " + info.name);
  }
  // 量化存储的 norm/bias:按 [行, 列] 读入后反量化,再还原成原形状。
  const int cols = shape.back();
  int elements = 1;
  for (int dimension : shape) elements *= dimension;
  const Tensor flattened =
      Dequantize(LoadBlocks(path, reader, info, elements / cols, cols));
  return flattened.Reshape(shape);
}

// 读取必需的 tensor,缺失时抛出异常。
const GgufTensorInfo& RequireTensor(const GgufReader& reader,
                                    const std::string& name) {
  const GgufTensorInfo* info = reader.FindTensor(name);
  if (info == nullptr) {
    throw std::runtime_error("GGUF 缺少必需的 tensor: " + name);
  }
  return *info;
}

// 读取可选 tensor,不存在时返回空张量。
Tensor OptionalF32(const std::string& path, const GgufReader& reader,
                   const std::string& name) {
  const GgufTensorInfo* info = reader.FindTensor(name);
  if (info == nullptr) return Tensor();
  return LoadF32(path, reader, *info);
}

// 读取整数 metadata,缺失时使用 fallback。
int Number(const GgufReader& reader, const std::string& key, int fallback) {
  return static_cast<int>(reader.Number(key, fallback));
}

// 从 metadata 构造模型配置。
ModelConfig BuildConfig(const GgufReader& reader) {
  ModelConfig config;
  const std::string arch = reader.String("general.architecture", "llama");

  config.n_layers = Number(reader, arch + ".block_count", 0);
  config.dim = Number(reader, arch + ".embedding_length", 0);
  config.hidden_dim = Number(reader, arch + ".feed_forward_length", 0);
  config.n_heads = Number(reader, arch + ".attention.head_count", 0);
  config.n_kv_heads =
      Number(reader, arch + ".attention.head_count_kv", config.n_heads);
  config.max_seq_len = Number(reader, arch + ".context_length", 0);
  config.rope_theta =
      static_cast<float>(reader.Number(arch + ".rope.freq_base", 10000.0));
  config.rms_norm_eps = static_cast<float>(
      reader.Number(arch + ".attention.layer_norm_rms_epsilon", 1e-5));
  if (arch == "qwen2") config.rope_type = RopeType::kNeoX;

  std::vector<std::string> tokens;
  if (reader.Strings("tokenizer.ggml.tokens", &tokens)) {
    config.vocab_size = static_cast<int>(tokens.size());
  }

  ValidateConfig(config);
  config.max_seq_len = std::min(config.max_seq_len, kMaxContextLength);
  return config;
}

// 检查量化权重的行列数。
void RequireWeightShape(const QuantizedTensor& weight, int rows, int cols,
                        const std::string& name) {
  if (weight.rows == rows && weight.cols == cols) return;
  throw std::runtime_error(
      "GGUF 权重形状不符: " + name + " 是 [" + std::to_string(weight.rows) +
      ", " + std::to_string(weight.cols) + "],期望 [" + std::to_string(rows) +
      ", " + std::to_string(cols) + "]");
}

// 检查一维张量的元素个数,空张量表示可选字段缺失。
void RequireVectorSize(const Tensor& tensor, size_t size,
                       const std::string& name) {
  if (tensor.Size() != 0 && tensor.Size() != size) {
    throw std::runtime_error("GGUF 张量长度不符: " + name);
  }
}

// 校验所有层权重的形状是否与配置一致。
void ValidateShapes(const Model& model) {
  const ModelConfig& config = model.config;
  const int dim = config.dim;
  const int attention = config.attention_dim();
  const int kv = config.kv_dim();

  RequireWeightShape(model.token_embedding, config.vocab_size, dim,
                     "token_embedding");
  RequireVectorSize(model.final_norm, static_cast<size_t>(dim), "final_norm");
  if (!model.tied_lm_head) {
    RequireWeightShape(model.lm_head, config.vocab_size, dim, "lm_head");
  }

  for (int layer = 0; layer < config.n_layers; ++layer) {
    const LayerWeights& weights = model.layers[static_cast<size_t>(layer)];
    const std::string prefix = "layers." + std::to_string(layer) + ".";
    RequireVectorSize(weights.attention_norm, static_cast<size_t>(dim),
                      prefix + "attention_norm");
    RequireWeightShape(weights.wq, attention, dim, prefix + "wq");
    RequireWeightShape(weights.wk, kv, dim, prefix + "wk");
    RequireWeightShape(weights.wv, kv, dim, prefix + "wv");
    RequireWeightShape(weights.wo, dim, attention, prefix + "wo");
    RequireVectorSize(weights.bq, static_cast<size_t>(attention),
                      prefix + "bq");
    RequireVectorSize(weights.bk, static_cast<size_t>(kv), prefix + "bk");
    RequireVectorSize(weights.bv, static_cast<size_t>(kv), prefix + "bv");
    RequireVectorSize(weights.ffn_norm, static_cast<size_t>(dim),
                      prefix + "ffn_norm");
    RequireWeightShape(weights.w_gate, config.hidden_dim, dim,
                       prefix + "w_gate");
    RequireWeightShape(weights.w_up, config.hidden_dim, dim, prefix + "w_up");
    RequireWeightShape(weights.w_down, dim, config.hidden_dim,
                       prefix + "w_down");
  }
}

}  // namespace

Model LoadGgufModel(const std::string& path) {
  GgufReader reader;
  if (!reader.Load(path)) {
    throw std::runtime_error("GGUF 解析失败: " + reader.Error());
  }

  Model model;
  model.config = BuildConfig(reader);
  const ModelConfig& config = model.config;

  model.token_embedding =
      LoadWeight(path, reader, RequireTensor(reader, "token_embd.weight"));
  model.final_norm =
      LoadF32(path, reader, RequireTensor(reader, "output_norm.weight"));

  model.layers.resize(static_cast<size_t>(config.n_layers));
  for (int layer = 0; layer < config.n_layers; ++layer) {
    LayerWeights& weights = model.layers[static_cast<size_t>(layer)];
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    weights.attention_norm = LoadF32(
        path, reader, RequireTensor(reader, prefix + "attn_norm.weight"));
    weights.wq = LoadWeight(path, reader,
                            RequireTensor(reader, prefix + "attn_q.weight"));
    weights.wk = LoadWeight(path, reader,
                            RequireTensor(reader, prefix + "attn_k.weight"));
    weights.wv = LoadWeight(path, reader,
                            RequireTensor(reader, prefix + "attn_v.weight"));
    weights.wo = LoadWeight(
        path, reader, RequireTensor(reader, prefix + "attn_output.weight"));
    weights.bq = OptionalF32(path, reader, prefix + "attn_q.bias");
    weights.bk = OptionalF32(path, reader, prefix + "attn_k.bias");
    weights.bv = OptionalF32(path, reader, prefix + "attn_v.bias");
    weights.ffn_norm = LoadF32(
        path, reader, RequireTensor(reader, prefix + "ffn_norm.weight"));
    weights.w_gate = LoadWeight(
        path, reader, RequireTensor(reader, prefix + "ffn_gate.weight"));
    weights.w_up = LoadWeight(
        path, reader, RequireTensor(reader, prefix + "ffn_up.weight"));
    weights.w_down = LoadWeight(
        path, reader, RequireTensor(reader, prefix + "ffn_down.weight"));
  }

  const GgufTensorInfo* output = reader.FindTensor("output.weight");
  if (output != nullptr) {
    model.lm_head = LoadWeight(path, reader, *output);
  } else {
    // 没有独立输出投影时与词表嵌入共享权重(Qwen2 等模型的常见做法)。
    model.tied_lm_head = true;
  }

  ValidateShapes(model);
  return model;
}

}  // namespace mini_lla
