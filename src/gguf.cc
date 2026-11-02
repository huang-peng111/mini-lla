#include "mini_llama/gguf.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "mini_llama/quantized_tensor.h"

namespace mini_lla {
namespace {

// 文本形式的 metadata 最多展示几个数组元素。
constexpr size_t kTextElementLimit = 4;

// 从二进制流读取一个小端字段,读不满时抛出异常。
template <typename T>
T Read(std::ifstream& file) {
  T value{};
  file.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!file) throw std::runtime_error("GGUF 文件提前结束");
  return value;
}

// 读取 GGUF 的长度前缀字符串。
std::string ReadString(std::ifstream& file) {
  const uint64_t length = Read<uint64_t>(file);
  if (length > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    throw std::runtime_error("GGUF 字符串过长");
  }
  std::string result(static_cast<size_t>(length), '\0');
  file.read(result.data(), static_cast<std::streamsize>(length));
  if (!file) throw std::runtime_error("GGUF 字符串数据不完整");
  return result;
}

// 读取一个定长的数值字段并统一转换成 double。
double ReadNumber(std::ifstream& file, GgufValueType type) {
  switch (type) {
    case GgufValueType::kUint8:
      return static_cast<double>(Read<uint8_t>(file));
    case GgufValueType::kInt8:
      return static_cast<double>(Read<int8_t>(file));
    case GgufValueType::kUint16:
      return static_cast<double>(Read<uint16_t>(file));
    case GgufValueType::kInt16:
      return static_cast<double>(Read<int16_t>(file));
    case GgufValueType::kUint32:
      return static_cast<double>(Read<uint32_t>(file));
    case GgufValueType::kInt32:
      return static_cast<double>(Read<int32_t>(file));
    case GgufValueType::kFloat32:
      return static_cast<double>(Read<float>(file));
    case GgufValueType::kBool:
      return Read<uint8_t>(file) != 0 ? 1.0 : 0.0;
    case GgufValueType::kUint64:
      return static_cast<double>(Read<uint64_t>(file));
    case GgufValueType::kInt64:
      return static_cast<double>(Read<int64_t>(file));
    case GgufValueType::kFloat64:
      return Read<double>(file);
    default:
      throw std::runtime_error("GGUF metadata 数值类型未知");
  }
}

// 把数组内容拼成便于打印的文本,过长的数组只展示前几个元素。
void FormatArrayText(const GgufMetadataKv& item, std::ostringstream* text) {
  *text << "[";
  const bool is_string = !item.strings.empty();
  const size_t count = is_string ? item.strings.size() : item.numbers.size();
  for (size_t i = 0; i < count; ++i) {
    if (i == kTextElementLimit) {
      *text << ", ...(" << count << ")";
      break;
    }
    if (i != 0) *text << ", ";
    if (is_string) {
      *text << item.strings[i];
    } else {
      *text << item.numbers[i];
    }
  }
  *text << "]";
}

// 读取一个 metadata 值,同时填充可读文本和结构化数据。
void ReadValue(std::ifstream& file, GgufValueType type,
               GgufMetadataKv* item) {
  std::ostringstream text;
  if (type == GgufValueType::kString) {
    item->value = ReadString(file);
    text << item->value;
  } else if (type == GgufValueType::kArray) {
    const auto element = static_cast<GgufValueType>(Read<uint32_t>(file));
    const uint64_t count = Read<uint64_t>(file);
    if (element == GgufValueType::kArray) {
      throw std::runtime_error("GGUF metadata 不支持嵌套数组");
    }
    for (uint64_t i = 0; i < count; ++i) {
      if (element == GgufValueType::kString) {
        item->strings.push_back(ReadString(file));
      } else {
        item->numbers.push_back(ReadNumber(file, element));
      }
    }
    FormatArrayText(*item, &text);
  } else {
    item->numbers.push_back(ReadNumber(file, type));
    text << item->numbers.front();
  }
  item->text = text.str();
}

// 按 GGUF 约定把偏移向上对齐,对齐值必须是 2 的幂。
uint64_t Align(uint64_t value, uint64_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    throw std::runtime_error("GGUF 对齐值必须是 2 的幂");
  }
  return (value + alignment - 1) & ~(alignment - 1);
}

// 读取 tensor 表的每一项。
void ReadTensorTable(std::ifstream& file, uint64_t count,
                     std::vector<GgufTensorInfo>* tensors) {
  for (uint64_t i = 0; i < count; ++i) {
    GgufTensorInfo info;
    info.name = ReadString(file);
    const uint32_t dimensions = Read<uint32_t>(file);
    if (dimensions == 0 || dimensions > 4) {
      throw std::runtime_error("GGUF tensor 维度数量不支持");
    }
    for (uint32_t d = 0; d < dimensions; ++d) {
      info.shape.push_back(Read<uint64_t>(file));
    }
    info.type = static_cast<GgmlType>(Read<uint32_t>(file));
    info.offset = Read<uint64_t>(file);
    tensors->push_back(std::move(info));
  }
}

}  // namespace

GgufReader::GgufReader(const std::string& path) {
  if (!Load(path)) throw std::runtime_error(error_);
}

bool GgufReader::Load(const std::string& path) {
  version_ = 0;
  tensor_count_ = 0;
  metadata_count_ = 0;
  data_offset_ = 0;
  metadata_.clear();
  tensors_.clear();
  error_.clear();
  try {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("无法打开 GGUF 文件: " + path);

    char magic[4]{};
    file.read(magic, sizeof(magic));
    if (std::memcmp(magic, "GGUF", 4) != 0) {
      throw std::runtime_error("GGUF magic 不正确");
    }

    version_ = Read<uint32_t>(file);
    tensor_count_ = Read<uint64_t>(file);
    metadata_count_ = Read<uint64_t>(file);
    if (version_ < 2 || version_ > 3) {
      throw std::runtime_error("只支持 GGUF v2/v3");
    }

    for (uint64_t i = 0; i < metadata_count_; ++i) {
      GgufMetadataKv item;
      item.key = ReadString(file);
      item.type = static_cast<GgufValueType>(Read<uint32_t>(file));
      ReadValue(file, item.type, &item);
      metadata_.push_back(std::move(item));
    }

    ReadTensorTable(file, tensor_count_, &tensors_);

    uint64_t alignment = 32;
    const GgufMetadataKv* alignment_item = FindMetadata("general.alignment");
    if (alignment_item != nullptr && !alignment_item->numbers.empty()) {
      alignment = static_cast<uint64_t>(alignment_item->numbers.front());
    }
    data_offset_ = Align(static_cast<uint64_t>(file.tellg()), alignment);
    return true;
  } catch (const std::exception& exception) {
    error_ = exception.what();
    return false;
  }
}

const GgufMetadataKv* GgufReader::FindMetadata(const std::string& key) const {
  for (const GgufMetadataKv& item : metadata_) {
    if (item.key == key) return &item;
  }
  return nullptr;
}

const GgufTensorInfo* GgufReader::FindTensor(const std::string& name) const {
  for (const GgufTensorInfo& item : tensors_) {
    if (item.name == name) return &item;
  }
  return nullptr;
}

std::string GgufReader::String(const std::string& key,
                               const std::string& fallback) const {
  const GgufMetadataKv* item = FindMetadata(key);
  return item != nullptr && !item->value.empty() ? item->value : fallback;
}

double GgufReader::Number(const std::string& key, double fallback) const {
  const GgufMetadataKv* item = FindMetadata(key);
  if (item == nullptr || item->numbers.empty()) return fallback;
  return item->numbers.front();
}

bool GgufReader::Strings(const std::string& key,
                         std::vector<std::string>* output) const {
  const GgufMetadataKv* item = FindMetadata(key);
  if (item == nullptr || item->strings.empty()) return false;
  *output = item->strings;
  return true;
}

bool GgufReader::Ints(const std::string& key,
                      std::vector<int32_t>* output) const {
  const GgufMetadataKv* item = FindMetadata(key);
  if (item == nullptr || item->numbers.empty()) return false;
  output->clear();
  output->reserve(item->numbers.size());
  for (double value : item->numbers) {
    output->push_back(static_cast<int32_t>(value));
  }
  return true;
}

const char* GgmlTypeName(GgmlType type) {
  switch (type) {
    case GgmlType::kF32:
      return "F32";
    case GgmlType::kF16:
      return "F16";
    case GgmlType::kQ4_0:
      return "Q4_0";
    case GgmlType::kQ4_1:
      return "Q4_1";
    case GgmlType::kQ8_0:
      return "Q8_0";
  }
  return "UNKNOWN";
}

bool GgufTensorDataSize(const GgufTensorInfo& info, uint64_t* size) {
  if (size == nullptr || info.shape.empty()) return false;
  uint64_t elements = 1;
  for (uint64_t dimension : info.shape) {
    if (dimension == 0 ||
        elements > std::numeric_limits<uint64_t>::max() / dimension) {
      return false;
    }
    elements *= dimension;
  }

  switch (info.type) {
    case GgmlType::kF32:
      *size = elements * 4;
      return true;
    case GgmlType::kF16:
      *size = elements * 2;
      return true;
    case GgmlType::kQ4_0:
    case GgmlType::kQ4_1:
    case GgmlType::kQ8_0:
      break;
  }

  const uint64_t block_size = static_cast<uint64_t>(kBlockSize);
  if (elements % block_size != 0) return false;
  const uint64_t blocks = elements / block_size;
  if (info.type == GgmlType::kQ8_0) *size = blocks * 34;
  if (info.type == GgmlType::kQ4_0) *size = blocks * 18;
  if (info.type == GgmlType::kQ4_1) *size = blocks * 20;
  return true;
}

bool ReadGgufTensorInto(const std::string& path, const GgufReader& reader,
                        const GgufTensorInfo& info, void* destination,
                        size_t bytes) {
  uint64_t expected = 0;
  if (!GgufTensorDataSize(info, &expected) || expected != bytes) {
    return false;
  }
  std::ifstream file(path, std::ios::binary);
  if (!file) return false;
  file.seekg(static_cast<std::streamoff>(reader.DataOffset() + info.offset));
  file.read(static_cast<char*>(destination), static_cast<std::streamsize>(bytes));
  return static_cast<bool>(file);
}

void InspectGguf(const std::string& path, const GgufReader& reader) {
  std::cout << "gguf: " << path << "\n"
            << "  version: " << reader.Version() << "\n"
            << "  tensors: " << reader.TensorCount() << "\n"
            << "  metadata: " << reader.MetadataCount() << "\n"
            << "  data offset: " << reader.DataOffset() << "\n\n";

  std::cout << "metadata:\n";
  for (const GgufMetadataKv& item : reader.Metadata()) {
    std::cout << "  " << item.key << " = " << item.text << "\n";
  }

  std::cout << "\ntensors:\n";
  for (const GgufTensorInfo& info : reader.Tensors()) {
    uint64_t bytes = 0;
    std::cout << "  " << info.name << " "
              << GgmlTypeName(info.type) << " [";
    for (size_t i = 0; i < info.shape.size(); ++i) {
      if (i != 0) std::cout << ", ";
      std::cout << info.shape[i];
    }
    std::cout << "]";
    if (GgufTensorDataSize(info, &bytes)) {
      std::cout << " " << bytes << " bytes";
    }
    std::cout << "\n";
  }
}

}  // namespace mini_lla
