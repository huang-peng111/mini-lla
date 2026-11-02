#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_GGUF_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_GGUF_H_

#include <cstdint>
#include <string>
#include <vector>

namespace mini_lla {

// GGML 权重类型,数值必须与 GGUF 格式定义保持一致。
enum class GgmlType : uint32_t {
  kF32 = 0,
  kF16 = 1,
  kQ4_0 = 2,
  kQ4_1 = 3,
  kQ8_0 = 8,
};

// GGUF metadata 的类型标签。
enum class GgufValueType : uint32_t {
  kUint8 = 0,
  kInt8 = 1,
  kUint16 = 2,
  kInt16 = 3,
  kUint32 = 4,
  kInt32 = 5,
  kFloat32 = 6,
  kBool = 7,
  kString = 8,
  kArray = 9,
  kUint64 = 10,
  kInt64 = 11,
  kFloat64 = 12,
};

// 一条 metadata:既保留可读文本,也把原始值整理成便于查询的形式。
struct GgufMetadataKv {
  std::string key;
  GgufValueType type = GgufValueType::kString;
  std::string text;                  // 打印用,长数组会被截断
  std::string value;                 // 字符串标量
  std::vector<std::string> strings;  // 字符串数组
  std::vector<double> numbers;       // 数值标量或数值数组
};

// tensor 描述:名称、形状、类型和数据区相对偏移。
// shape 保持 GGUF 顺序,即第一个维度变化最快。
struct GgufTensorInfo {
  std::string name;
  std::vector<uint64_t> shape;
  GgmlType type = GgmlType::kF32;
  uint64_t offset = 0;
};

// GGUF 读取器:解析文件头、metadata 和 tensor 表,并计算数据区绝对偏移。
class GgufReader {
 public:
  GgufReader() = default;

  // 打开并解析文件,失败时抛出异常。
  explicit GgufReader(const std::string& path);

  // 打开并解析文件,失败返回 false,原因写在 Error() 中。
  bool Load(const std::string& path);

  uint32_t Version() const { return version_; }
  uint64_t TensorCount() const { return tensor_count_; }
  uint64_t MetadataCount() const { return metadata_count_; }
  const std::vector<GgufMetadataKv>& Metadata() const { return metadata_; }
  const std::vector<GgufTensorInfo>& Tensors() const { return tensors_; }

  // 返回 tensor 数据区起始的绝对文件偏移。
  uint64_t DataOffset() const { return data_offset_; }

  // 返回最后一次 Load 的错误原因。
  const std::string& Error() const { return error_; }

  // 按 key 查找 metadata,找不到返回 nullptr。
  const GgufMetadataKv* FindMetadata(const std::string& key) const;

  // 按名称查找 tensor,找不到返回 nullptr。
  const GgufTensorInfo* FindTensor(const std::string& name) const;

  // 读取字符串标量,键不存在或类型不符时返回 fallback。
  std::string String(const std::string& key,
                     const std::string& fallback) const;

  // 读取数值标量,键不存在或类型不符时返回 fallback。
  double Number(const std::string& key, double fallback) const;

  // 读取字符串数组,键不存在时返回 false。
  bool Strings(const std::string& key, std::vector<std::string>* output) const;

  // 读取整数数组,键不存在时返回 false。
  bool Ints(const std::string& key, std::vector<int32_t>* output) const;

 private:
  uint32_t version_ = 0;
  uint64_t tensor_count_ = 0;
  uint64_t metadata_count_ = 0;
  uint64_t data_offset_ = 0;
  std::vector<GgufMetadataKv> metadata_;
  std::vector<GgufTensorInfo> tensors_;
  std::string error_;
};

// 返回 GGML 类型名称,例如 "Q8_0"。
const char* GgmlTypeName(GgmlType type);

// 计算 tensor 数据占用的字节数,类型不支持时返回 false。
bool GgufTensorDataSize(const GgufTensorInfo& info, uint64_t* size);

// 把 tensor 数据读入调用者提供的缓冲区,避免额外的临时副本。
// destination 至少要有 bytes 字节,bytes 必须等于 tensor 的数据大小。
bool ReadGgufTensorInto(const std::string& path, const GgufReader& reader,
                        const GgufTensorInfo& info, void* destination,
                        size_t bytes);

// 打印 GGUF 概况,供 inspect-gguf 命令使用。
void InspectGguf(const std::string& path, const GgufReader& reader);

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_GGUF_H_
