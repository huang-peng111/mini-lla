#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_GGUF_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_GGUF_H_
#include <cstdint>
#include <string>
#include <vector>
namespace mini_lla {
// GGUF 文件头中的基础信息。
struct GgufHeader { uint32_t version = 0; uint64_t tensor_count = 0; uint64_t metadata_count = 0; };
// GGUF tensor 的名称、形状、类型和数据偏移。
struct GgufTensorInfo { std::string name; std::vector<uint64_t> shape; uint32_t type = 0; uint64_t offset = 0; };
// 读取 GGUF 文件头和 tensor 描述。
class GgufReader {
 public:
  // 打开并解析指定 GGUF 文件。
  explicit GgufReader(const std::string& path);
  // 返回文件头信息。
  const GgufHeader& header() const { return header_; }
  // 返回全部 tensor 描述。
  const std::vector<GgufTensorInfo>& tensors() const { return tensors_; }
 private:
  GgufHeader header_;
  std::vector<GgufTensorInfo> tensors_;
};
}  // namespace mini_lla
#endif
