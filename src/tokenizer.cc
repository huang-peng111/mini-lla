#include "mini_llama/tokenizer.h"

namespace mini_lla {

std::vector<int> AsciiTokenizer::Encode(const std::string& text) const {
  std::vector<int> tokens{bos_id()};
  for (unsigned char value : text) {
    tokens.push_back(value < 128 ? value : unk_id());
  }
  return tokens;
}

std::string AsciiTokenizer::Decode(const std::vector<int>& tokens) const {
  std::string text;
  for (int token : tokens) {
    text += DecodeToken(token);
  }
  return text;
}

std::string AsciiTokenizer::DecodeToken(int token) const {
  if (token < 32 || token >= 128) {
    return "";
  }
  return std::string(1, static_cast<char>(token));
}

std::unique_ptr<ITokenizer> CreateAsciiTokenizer() {
  return std::make_unique<AsciiTokenizer>();
}

}  // namespace mini_lla
