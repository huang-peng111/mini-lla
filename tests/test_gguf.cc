#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "gguf_fixture.h"
#include "mini_llama/gguf.h"
#include "test_main.h"

using namespace mini_lla;

namespace {

// 写入一个只有头部的 GGUF 文件,用于错误路径测试。
void WriteGgufHeader(const std::string& path, const char* magic,
                     uint32_t version, uint64_t tensors, uint64_t metadata) {
  std::ofstream output(path, std::ios::binary);
  output.write(magic, 4);
  output.write(reinterpret_cast<const char*>(&version), sizeof(version));
  output.write(reinterpret_cast<const char*>(&tensors), sizeof(tensors));
  output.write(reinterpret_cast<const char*>(&metadata), sizeof(metadata));
}

}  // namespace

// 覆盖 GGUF reader 的 header、metadata、tensor 表和错误路径。
void TestGguf() {
  const std::string path = test::TempPath("mini_lla_reader.gguf");
  test::WriteFixtureGguf(path);

  GgufReader reader;
  CHECK_TRUE(reader.Load(path));
  CHECK_EQUAL(reader.Version(), 3);
  CHECK_EQUAL(reader.TensorCount(), 14);
  CHECK_EQUAL(reader.MetadataCount(), 17);
  CHECK_TRUE(reader.DataOffset() > 0);

  // 类型化的 metadata 查询。
  CHECK_TRUE(reader.String("general.architecture", "") == "qwen2");
  CHECK_NEAR(reader.Number("qwen2.block_count", 0.0), 1.0, 1e-9);
  CHECK_NEAR(reader.Number("qwen2.attention.head_count_kv", 0.0), 1.0, 1e-9);
  CHECK_TRUE(reader.String("no.such.key", "fallback") == "fallback");
  CHECK_NEAR(reader.Number("no.such.key", -1.0), -1.0, 1e-9);

  std::vector<std::string> tokens;
  CHECK_TRUE(reader.Strings("tokenizer.ggml.tokens", &tokens));
  CHECK_EQUAL(tokens.size(), 10u);
  CHECK_TRUE(tokens[1] == "<|im_start|>");
  std::vector<int32_t> types;
  CHECK_TRUE(reader.Ints("tokenizer.ggml.token_type", &types));
  CHECK_EQUAL(types.size(), 10u);
  CHECK_EQUAL(types[9], 6);
  CHECK_TRUE(!reader.Strings("general.architecture", &tokens));

  // tensor 表:形状保持 GGUF 顺序,Q8_0 的字节数按 block 计算。
  const GgufTensorInfo* attention_q = reader.FindTensor("blk.0.attn_q.weight");
  CHECK_TRUE(attention_q != nullptr);
  CHECK_EQUAL(attention_q->shape[0], 32u);
  CHECK_EQUAL(attention_q->shape[1], 32u);
  CHECK_TRUE(attention_q->type == GgmlType::kQ8_0);
  CHECK_TRUE(reader.FindTensor("no.such.tensor") == nullptr);

  uint64_t bytes = 0;
  CHECK_TRUE(GgufTensorDataSize(*attention_q, &bytes));
  CHECK_EQUAL(bytes, 1088u);

  std::vector<uint8_t> buffer(static_cast<size_t>(bytes));
  CHECK_TRUE(ReadGgufTensorInto(path, reader, *attention_q, buffer.data(),
                                buffer.size()));
  CHECK_TRUE(
      !ReadGgufTensorInto(path, reader, *attention_q, buffer.data(), 10));

  // 错误路径:magic、版本、截断文件都要被拒绝,并给出原因。
  const std::string bad_magic = test::TempPath("mini_lla_bad_magic.gguf");
  WriteGgufHeader(bad_magic, "XXXX", 3, 0, 0);
  GgufReader magic_reader;
  CHECK_TRUE(!magic_reader.Load(bad_magic));
  CHECK_TRUE(!magic_reader.Error().empty());

  const std::string bad_version = test::TempPath("mini_lla_bad_version.gguf");
  WriteGgufHeader(bad_version, "GGUF", 1, 0, 0);
  GgufReader version_reader;
  CHECK_TRUE(!version_reader.Load(bad_version));

  const std::string truncated = test::TempPath("mini_lla_truncated.gguf");
  WriteGgufHeader(truncated, "GGUF", 3, 5, 0);
  GgufReader truncated_reader;
  CHECK_TRUE(!truncated_reader.Load(truncated));
  CHECK_TRUE(!truncated_reader.Error().empty());
}
