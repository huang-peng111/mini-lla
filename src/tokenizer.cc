#include "mini_llama/tokenizer.h"
namespace mini_lla {
// 将输入文本编码为 BOS 加 ASCII token 序列。
std::vector<int> AsciiTokenizer::encode(const std::string& text) const {
  // 每个 prompt 都以 BOS 开始，方便后续替换成更完整的 tokenizer。
  // 每个 prompt 都以 BOS 开始，方便后续替换成更完整的 tokenizer。
  std::vector<int> tokens{bos_id()};
  for (unsigned char value : text)
    tokens.push_back(value < 128 ? value : unk_id());
  return tokens;
}
// 将单个 token id 解码为可显示字符。
std::string AsciiTokenizer::decode_token(int token) const {
  // 控制 token 不直接打印，普通 ASCII token 还原为单字符。
  // 控制 token 不直接打印，普通 ASCII token 还原为单字符。
  if (token == bos_id() || token == eos_id() || token == unk_id() ||
      token < 32 || token >= 128)
    return "";
  return std::string(1, static_cast<char>(token));
}
}  // namespace mini_lla
