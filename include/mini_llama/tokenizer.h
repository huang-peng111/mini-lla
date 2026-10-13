#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_TOKENIZER_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_TOKENIZER_H_
#include <string>
#include <vector>
namespace mini_lla {
// 教学版字符 tokenizer：一个 ASCII 字符对应一个 token id。
class AsciiTokenizer {
 public:
  // 将文本转换为 BOS 加字符 token 序列。
  std::vector<int> encode(const std::string& text) const;
  // 将一个 token id 转换为可显示文本。
  std::string decode_token(int token) const;
  // 返回词表大小。
  int vocab_size() const { return 128; }
  // 返回起始 token id。
  int bos_id() const { return 1; }
  // 返回结束 token id。
  int eos_id() const { return 2; }
  // 返回未知 token id。
  int unk_id() const { return 0; }
};
}  // namespace mini_lla
#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_TOKENIZER_H_
