#ifndef MINI_LLA_TESTS_GGUF_FIXTURE_H_
#define MINI_LLA_TESTS_GGUF_FIXTURE_H_

#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "mini_llama/quant.h"
#include "mini_llama/quantized_tensor.h"

namespace mini_lla {
namespace test {

// 测试用 GGUF 的 tensor 描述。
// shape 使用逻辑形状 [输出, 输入],写入文件时反转成 GGUF 的 ne0 在前的顺序。
struct FixtureTensor {
  std::string name;
  std::vector<int> shape;
  std::vector<float> values;
  QuantType type = QuantType::kF32;
};

// 测试用 metadata 的取值类型。
enum class FixtureValue {
  kString,
  kUint32,
  kFloat32,
  kBool,
  kStringArray,
  kInt32Array,
};

// 测试用 GGUF 的一条 metadata。
struct FixtureMetadata {
  std::string key;
  FixtureValue kind = FixtureValue::kString;
  std::string text;
  std::vector<std::string> strings;
  std::vector<int32_t> ints;
  double number = 0.0;
};

// 把整数向上对齐到 32 字节,模拟 GGUF 的数据对齐。
inline uint64_t Align32(uint64_t value) {
  return (value + 31) & ~static_cast<uint64_t>(31);
}

// 写入一个小端 POD 值。
template <typename T>
void WriteRaw(std::ofstream& file, const T& value) {
  file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

// 写入 GGUF 的长度前缀字符串。
inline void WriteText(std::ofstream& file, const std::string& text) {
  WriteRaw<uint64_t>(file, static_cast<uint64_t>(text.size()));
  file.write(text.data(), static_cast<std::streamsize>(text.size()));
}

// 返回当前写入位置。
inline uint64_t Position(std::ofstream& file) {
  return static_cast<uint64_t>(static_cast<std::streamoff>(file.tellp()));
}

// 返回量化类型对应的 GGUF 类型编号。
inline uint32_t GgufTypeValue(QuantType type) {
  switch (type) {
    case QuantType::kF32:
      return 0;
    case QuantType::kQ8_0:
      return 8;
    case QuantType::kQ4_0:
      return 2;
    case QuantType::kQ4_1:
      return 3;
  }
  return 0;
}

// 返回 tensor 数据的字节数。
inline uint64_t FixtureTensorBytes(const FixtureTensor& tensor) {
  if (tensor.type == QuantType::kF32) {
    return tensor.values.size() * sizeof(float);
  }
  const size_t blocks = tensor.values.size() / static_cast<size_t>(kBlockSize);
  if (tensor.type == QuantType::kQ8_0) return blocks * sizeof(BlockQ8_0);
  if (tensor.type == QuantType::kQ4_0) return blocks * sizeof(BlockQ4_0);
  return blocks * sizeof(BlockQ4_1);
}

// 写入一个 tensor 的数据:量化类型先量化再写 block。
inline void WriteTensorData(std::ofstream& file, const FixtureTensor& tensor) {
  if (tensor.type == QuantType::kF32) {
    file.write(reinterpret_cast<const char*>(tensor.values.data()),
               static_cast<std::streamsize>(tensor.values.size() *
                                            sizeof(float)));
    return;
  }

  Tensor matrix({tensor.shape[0], tensor.shape[1]});
  matrix.data = tensor.values;
  const QuantizedTensor weight = Quantize(matrix, tensor.type);
  switch (weight.type) {
    case QuantType::kQ8_0:
      file.write(reinterpret_cast<const char*>(weight.q8_0.data()),
                 static_cast<std::streamsize>(weight.q8_0.size() *
                                              sizeof(BlockQ8_0)));
      return;
    case QuantType::kQ4_0:
      file.write(reinterpret_cast<const char*>(weight.q4_0.data()),
                 static_cast<std::streamsize>(weight.q4_0.size() *
                                              sizeof(BlockQ4_0)));
      return;
    case QuantType::kQ4_1:
      file.write(reinterpret_cast<const char*>(weight.q4_1.data()),
                 static_cast<std::streamsize>(weight.q4_1.size() *
                                              sizeof(BlockQ4_1)));
      return;
    case QuantType::kF32:
      return;
  }
}

// 写入一条 metadata:先写类型标签,再写值。
inline void WriteMetadata(std::ofstream& file, const FixtureMetadata& item) {
  WriteText(file, item.key);
  switch (item.kind) {
    case FixtureValue::kString:
      WriteRaw<uint32_t>(file, 8);
      WriteText(file, item.text);
      return;
    case FixtureValue::kUint32:
      WriteRaw<uint32_t>(file, 4);
      WriteRaw<uint32_t>(file, static_cast<uint32_t>(item.number));
      return;
    case FixtureValue::kFloat32:
      WriteRaw<uint32_t>(file, 6);
      WriteRaw<float>(file, static_cast<float>(item.number));
      return;
    case FixtureValue::kBool:
      WriteRaw<uint32_t>(file, 7);
      WriteRaw<uint8_t>(file, item.number != 0.0 ? 1 : 0);
      return;
    case FixtureValue::kStringArray:
      WriteRaw<uint32_t>(file, 9);
      WriteRaw<uint32_t>(file, 8);
      WriteRaw<uint64_t>(file, static_cast<uint64_t>(item.strings.size()));
      for (const std::string& text : item.strings) WriteText(file, text);
      return;
    case FixtureValue::kInt32Array:
      WriteRaw<uint32_t>(file, 9);
      WriteRaw<uint32_t>(file, 5);
      WriteRaw<uint64_t>(file, static_cast<uint64_t>(item.ints.size()));
      for (int32_t value : item.ints) WriteRaw<int32_t>(file, value);
      return;
  }
}

// 生成确定的小权重,数值有限且各不相同。
inline std::vector<float> MakeValues(size_t count, float scale) {
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i) {
    values[i] = scale * std::sin(static_cast<float>(i) * 0.7f);
  }
  return values;
}

// 生成全 1 的向量,用于 norm 权重。
inline std::vector<float> MakeOnes(size_t count) {
  return std::vector<float>(count, 1.0f);
}

// 构造 qwen2 单层模型需要的 tensor 列表。
// dim=32、hidden=64、vocab=10、n_heads=2、n_kv_heads=1(GQA)。
inline std::vector<FixtureTensor> BuildFixtureTensors() {
  const int dim = 32;
  const int hidden = 64;
  const int vocab = 10;
  const int kv_dim = 16;
  return {
      {"token_embd.weight", {vocab, dim}, MakeValues(vocab * dim, 0.05f),
       QuantType::kQ8_0},
      {"output_norm.weight", {dim}, MakeOnes(dim), QuantType::kF32},
      {"blk.0.attn_norm.weight", {dim}, MakeOnes(dim), QuantType::kF32},
      {"blk.0.attn_q.weight", {dim, dim}, MakeValues(dim * dim, 0.05f),
       QuantType::kQ8_0},
      {"blk.0.attn_q.bias", {dim}, MakeValues(dim, 0.01f), QuantType::kF32},
      {"blk.0.attn_k.weight", {kv_dim, dim}, MakeValues(kv_dim * dim, 0.05f),
       QuantType::kQ8_0},
      {"blk.0.attn_k.bias", {kv_dim}, MakeValues(kv_dim, 0.01f),
       QuantType::kF32},
      {"blk.0.attn_v.weight", {kv_dim, dim}, MakeValues(kv_dim * dim, 0.05f),
       QuantType::kQ4_0},
      {"blk.0.attn_v.bias", {kv_dim}, MakeValues(kv_dim, 0.01f),
       QuantType::kF32},
      {"blk.0.attn_output.weight", {dim, dim}, MakeValues(dim * dim, 0.05f),
       QuantType::kQ8_0},
      {"blk.0.ffn_norm.weight", {dim}, MakeOnes(dim), QuantType::kF32},
      {"blk.0.ffn_gate.weight", {hidden, dim}, MakeValues(hidden * dim, 0.05f),
       QuantType::kQ8_0},
      {"blk.0.ffn_up.weight", {hidden, dim}, MakeValues(hidden * dim, 0.05f),
       QuantType::kQ4_1},
      {"blk.0.ffn_down.weight", {dim, hidden}, MakeValues(dim * hidden, 0.05f),
       QuantType::kQ8_0},
  };
}

// 构造测试用 metadata:模型超参、词表和两条 BPE 合并规则。
inline std::vector<FixtureMetadata> BuildFixtureMetadata() {
  // "\xC4\xA0" 是 GPT-2 字节映射里空格对应的 unicode 字符。
  const std::string space = "\xC4\xA0";
  return {
      {"general.architecture", FixtureValue::kString, "qwen2", {}, {}, 0.0},
      {"qwen2.block_count", FixtureValue::kUint32, "", {}, {}, 1.0},
      {"qwen2.embedding_length", FixtureValue::kUint32, "", {}, {}, 32.0},
      {"qwen2.feed_forward_length", FixtureValue::kUint32, "", {}, {}, 64.0},
      {"qwen2.attention.head_count", FixtureValue::kUint32, "", {}, {}, 2.0},
      {"qwen2.attention.head_count_kv", FixtureValue::kUint32, "", {}, {},
       1.0},
      {"qwen2.context_length", FixtureValue::kUint32, "", {}, {}, 32.0},
      {"qwen2.rope.freq_base", FixtureValue::kFloat32, "", {}, {}, 10000.0},
      {"qwen2.attention.layer_norm_rms_epsilon", FixtureValue::kFloat32, "",
       {}, {}, 1e-05},
      {"tokenizer.ggml.tokens",
       FixtureValue::kStringArray,
       "",
       {"<unk>", "<|im_start|>", "<|im_end|>", "h", "i", "hi", space,
        space + "h", "!", "<0x7A>"},
       {},
       0.0},
      {"tokenizer.ggml.token_type",
       FixtureValue::kInt32Array,
       "",
       {},
       {3, 3, 3, 1, 1, 1, 1, 1, 1, 6},
       0.0},
      {"tokenizer.ggml.merges",
       FixtureValue::kStringArray,
       "",
       {"h i", space + " h"},
       {},
       0.0},
      {"tokenizer.ggml.bos_token_id", FixtureValue::kUint32, "", {}, {}, 0.0},
      {"tokenizer.ggml.eos_token_id", FixtureValue::kUint32, "", {}, {}, 2.0},
      {"tokenizer.ggml.unk_token_id", FixtureValue::kUint32, "", {}, {}, 0.0},
      {"tokenizer.ggml.add_bos_token", FixtureValue::kBool, "", {}, {}, 0.0},
      {"tokenizer.chat_template", FixtureValue::kString,
       "{{ messages }}", {}, {}, 0.0},
  };
}

// 生成一个完整的 GGUF 测试文件,返回文件路径。
// include_tensors 为 false 时只写 metadata,用于测试缺失 tensor 的错误路径。
inline std::string WriteFixtureGguf(const std::string& path,
                                    bool include_tensors = true) {
  const std::vector<FixtureMetadata> metadata = BuildFixtureMetadata();
  const std::vector<FixtureTensor> tensors =
      include_tensors ? BuildFixtureTensors() : std::vector<FixtureTensor>();

  // 先算出每个 tensor 在数据区中的偏移,每项按 32 字节对齐。
  std::vector<uint64_t> offsets;
  uint64_t offset = 0;
  for (const FixtureTensor& tensor : tensors) {
    offset = Align32(offset);
    offsets.push_back(offset);
    offset += FixtureTensorBytes(tensor);
  }

  std::ofstream file(path, std::ios::binary);
  file.write("GGUF", 4);
  WriteRaw<uint32_t>(file, 3);
  WriteRaw<uint64_t>(file, static_cast<uint64_t>(tensors.size()));
  WriteRaw<uint64_t>(file, static_cast<uint64_t>(metadata.size()));
  for (const FixtureMetadata& item : metadata) WriteMetadata(file, item);

  for (size_t i = 0; i < tensors.size(); ++i) {
    WriteText(file, tensors[i].name);
    const std::vector<uint64_t> gguf_shape(tensors[i].shape.rbegin(),
                                           tensors[i].shape.rend());
    WriteRaw<uint32_t>(file, static_cast<uint32_t>(gguf_shape.size()));
    for (uint64_t dimension : gguf_shape) WriteRaw<uint64_t>(file, dimension);
    WriteRaw<uint32_t>(file, GgufTypeValue(tensors[i].type));
    WriteRaw<uint64_t>(file, offsets[i]);
  }

  const uint64_t data_start = Align32(Position(file));
  for (size_t i = 0; i < tensors.size(); ++i) {
    while (Position(file) < data_start + offsets[i]) WriteRaw<uint8_t>(file, 0);
    WriteTensorData(file, tensors[i]);
  }
  file.close();
  return path;
}

}  // namespace test
}  // namespace mini_lla

#endif  // MINI_LLA_TESTS_GGUF_FIXTURE_H_
