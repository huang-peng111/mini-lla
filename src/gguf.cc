#include "mini_llama/gguf.h"
#include <cstring>
#include <fstream>
#include <stdexcept>
namespace mini_lla {
namespace {
// 从二进制流读取固定大小的基础类型。
template <typename T> T read(std::ifstream& file) { T value{}; file.read(reinterpret_cast<char*>(&value), sizeof(value)); if (!file) throw std::runtime_error("truncated GGUF file"); return value; }
}
GgufReader::GgufReader(const std::string& path) {
  std::ifstream file(path, std::ios::binary); if (!file) throw std::runtime_error("cannot open GGUF: " + path);
  char magic[4]; file.read(magic, 4); if (std::memcmp(magic, "GGUF", 4) != 0) throw std::runtime_error("invalid GGUF magic");
  header_.version = read<uint32_t>(file); header_.tensor_count = read<uint64_t>(file); header_.metadata_count = read<uint64_t>(file);
  if (header_.version < 1 || header_.version > 3) throw std::runtime_error("unsupported GGUF version");
  // 第一版先验证头部；metadata 和 tensor table 在后续 loader 阶段扩展。
}
}  // namespace mini_lla
