#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_GGUF_TOKENIZER_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_GGUF_TOKENIZER_H_
#include <string>
#include <unordered_map>
#include <vector>
namespace mini_lla {
// 简化的词表 tokenizer，使用最长字符串匹配模拟 BPE 输入。
class VocabularyTokenizer {
 public:
  // 加载 token 文本到 id 的词表。
  bool load(const std::string& path);
  // 将文本编码为 token id。
  std::vector<int> encode(const std::string& text) const;
  // 将 token id 解码为文本。
  std::string decode(const std::vector<int>& tokens) const;
 private:
  std::unordered_map<std::string, int> token_to_id_;
  std::unordered_map<int, std::string> id_to_token_;
};
}  // namespace mini_lla
#endif
