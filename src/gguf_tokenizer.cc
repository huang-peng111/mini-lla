#include "mini_llama/gguf_tokenizer.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "mini_llama/gguf.h"

namespace mini_lla {
namespace {

// GGUF 中 token 类型的取值。
constexpr int32_t kTokenTypeControl = 3;
constexpr int32_t kTokenTypeUserDefined = 4;
constexpr int32_t kTokenTypeByte = 6;

// 把 unicode 码点编码成 UTF-8 字符串。
std::string CodepointToUtf8(char32_t code) {
  std::string result;
  if (code <= 0x7F) {
    result += static_cast<char>(code);
  } else if (code <= 0x7FF) {
    result += static_cast<char>(0xC0 | (code >> 6));
    result += static_cast<char>(0x80 | (code & 0x3F));
  } else if (code <= 0xFFFF) {
    result += static_cast<char>(0xE0 | (code >> 12));
    result += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
    result += static_cast<char>(0x80 | (code & 0x3F));
  } else {
    result += static_cast<char>(0xF0 | (code >> 18));
    result += static_cast<char>(0x80 | ((code >> 12) & 0x3F));
    result += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
    result += static_cast<char>(0x80 | (code & 0x3F));
  }
  return result;
}

// 把字符串按 UTF-8 拆成单个字符,非法字节会被跳过。
std::vector<std::string> SplitUtf8(const std::string& text) {
  std::vector<std::string> characters;
  size_t position = 0;
  while (position < text.size()) {
    const unsigned char first = static_cast<unsigned char>(text[position]);
    size_t length = 1;
    if ((first & 0xE0) == 0xC0) length = 2;
    if ((first & 0xF0) == 0xE0) length = 3;
    if ((first & 0xF8) == 0xF0) length = 4;
    if (position + length > text.size()) {
      ++position;
      continue;
    }
    characters.push_back(text.substr(position, length));
    position += length;
  }
  return characters;
}

// 判断字符是否属于空白,用于切分片段。
bool IsSpace(char value) {
  return value == ' ' || value == '\t' || value == '\n' || value == '\r';
}

// 把文本切成片段:每个空格开始一个新片段,首字符是空格时除外。
// 这样 "hello world" 会切成 "hello" 和 " world",与 GPT-2 的切分接近。
std::vector<std::string> SplitSegments(const std::string& text) {
  std::vector<std::string> segments;
  size_t start = 0;
  for (size_t i = 1; i < text.size(); ++i) {
    if (IsSpace(text[i]) && !IsSpace(text[i - 1])) {
      segments.push_back(text.substr(start, i - start));
      start = i;
    }
  }
  if (start < text.size()) {
    segments.push_back(text.substr(start));
  }
  return segments;
}

// 解析字节 token 的文本形式 "<0x0A>",不是该形式时返回 -1。
int ParseByteToken(const std::string& text) {
  if (text.size() != 6 || text.compare(0, 3, "<0x") != 0 || text[5] != '>') {
    return -1;
  }
  int value = 0;
  std::istringstream hex(text.substr(3, 2));
  if (!(hex >> std::hex >> value) || value < 0 || value > 255) {
    return -1;
  }
  return value;
}

}  // namespace

void GgufTokenizer::BuildByteMap() {
  // 可打印字节直接映射到自身,其余字节映射到 256 以上的码点。
  std::vector<int> bytes;
  for (int c = '!'; c <= '~'; ++c) bytes.push_back(c);
  for (int c = 0xA1; c <= 0xAC; ++c) bytes.push_back(c);
  for (int c = 0xAE; c <= 0xFF; ++c) bytes.push_back(c);

  std::vector<int> codes = bytes;
  int extra = 0;
  for (int value = 0; value < 256; ++value) {
    if (std::find(bytes.begin(), bytes.end(), value) == bytes.end()) {
      bytes.push_back(value);
      codes.push_back(256 + extra);
      ++extra;
    }
  }

  byte_to_char_.assign(256, std::string());
  char_to_byte_.clear();
  for (size_t i = 0; i < bytes.size(); ++i) {
    const std::string character = CodepointToUtf8(static_cast<char32_t>(codes[i]));
    byte_to_char_[static_cast<size_t>(bytes[i])] = character;
    char_to_byte_[character] = static_cast<uint8_t>(bytes[i]);
  }
}

bool GgufTokenizer::LoadVocabulary(const std::string& path) {
  GgufReader reader;
  if (!reader.Load(path)) return false;
  if (!reader.Strings("tokenizer.ggml.tokens", &tokens_)) return false;
  if (tokens_.empty()) return false;

  for (size_t i = 0; i < tokens_.size(); ++i) {
    token_ids_[tokens_[i]] = static_cast<int>(i);
  }

  std::vector<std::string> merges;
  if (reader.Strings("tokenizer.ggml.merges", &merges)) {
    int rank = 0;
    for (const std::string& merge : merges) {
      std::istringstream stream(merge);
      std::string left;
      std::string right;
      if (stream >> left >> right) {
        merge_ranks_[{left, right}] = rank++;
      }
    }
  }

  std::vector<int32_t> types;
  token_bytes_.assign(tokens_.size(), -1);
  byte_token_ids_.assign(256, -1);
  special_tokens_.clear();
  if (reader.Ints("tokenizer.ggml.token_type", &types) &&
      types.size() == tokens_.size()) {
    ClassifyTokens(types);
  }

  bos_id_ = static_cast<int>(
      reader.Number("tokenizer.ggml.bos_token_id", -1));
  eos_id_ = static_cast<int>(
      reader.Number("tokenizer.ggml.eos_token_id", -1));
  unk_id_ = static_cast<int>(
      reader.Number("tokenizer.ggml.unk_token_id", -1));
  add_bos_ = reader.Number("tokenizer.ggml.add_bos_token", 0.0) != 0.0;
  BuildByteMap();
  return true;
}

void GgufTokenizer::ClassifyTokens(const std::vector<int32_t>& types) {
  for (size_t i = 0; i < tokens_.size(); ++i) {
    if (types[i] == kTokenTypeControl || types[i] == kTokenTypeUserDefined) {
      special_tokens_.emplace_back(tokens_[i], static_cast<int>(i));
      continue;
    }
    if (types[i] != kTokenTypeByte) continue;
    const int value = ParseByteToken(tokens_[i]);
    if (value >= 0) {
      byte_token_ids_[static_cast<size_t>(value)] = static_cast<int>(i);
      token_bytes_[i] = value;
    }
  }

  std::sort(special_tokens_.begin(), special_tokens_.end(),
            [](const auto& left, const auto& right) {
              return left.first.size() > right.first.size();
            });
}

bool GgufTokenizer::Load(const std::string& path) {
  return LoadVocabulary(path);
}

void GgufTokenizer::AppendByteFallback(const std::string& piece,
                                       std::vector<int>* output) const {
  for (const std::string& character : SplitUtf8(piece)) {
    const auto found = char_to_byte_.find(character);
    const int byte = found == char_to_byte_.end() ? -1 : found->second;
    const int fallback = byte >= 0 ? byte_token_ids_[static_cast<size_t>(byte)]
                                   : -1;
    if (fallback >= 0) {
      output->push_back(fallback);
    } else if (unk_id_ >= 0) {
      output->push_back(unk_id_);
    }
  }
}

void GgufTokenizer::EncodeSegment(const std::string& segment,
                                  std::vector<int>* output) const {
  // 1. 每个字节先映射成一个 unicode 字符,得到初始符号序列。
  std::vector<std::string> symbols;
  symbols.reserve(segment.size());
  for (unsigned char value : segment) {
    symbols.push_back(byte_to_char_[value]);
  }

  // 2. 反复合并 rank 最小的相邻符号。
  while (symbols.size() > 1) {
    size_t best_index = symbols.size();
    int best_rank = std::numeric_limits<int>::max();
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      const auto found = merge_ranks_.find({symbols[i], symbols[i + 1]});
      if (found != merge_ranks_.end() && found->second < best_rank) {
        best_rank = found->second;
        best_index = i;
      }
    }
    if (best_index == symbols.size()) break;
    symbols[best_index] += symbols[best_index + 1];
    symbols.erase(symbols.begin() + static_cast<long>(best_index) + 1);
  }

  // 3. 查表得到 token id,片段不在词表时按字节回退。
  for (const std::string& symbol : symbols) {
    const auto found = token_ids_.find(symbol);
    if (found != token_ids_.end()) {
      output->push_back(found->second);
    } else {
      AppendByteFallback(symbol, output);
    }
  }
}

std::vector<int> GgufTokenizer::Encode(const std::string& text) const {
  std::vector<int> output;
  if (add_bos_ && bos_id_ >= 0) {
    output.push_back(bos_id_);
  }

  size_t position = 0;
  while (position < text.size()) {
    // 优先匹配特殊 token,保证 "<|im_start|>" 不会被拆开。
    bool matched = false;
    for (const auto& special : special_tokens_) {
      if (text.compare(position, special.first.size(), special.first) == 0) {
        output.push_back(special.second);
        position += special.first.size();
        matched = true;
        break;
      }
    }
    if (matched) continue;

    // 找到下一个特殊 token 的位置,中间的普通文本按片段编码。
    size_t next = text.size();
    for (const auto& special : special_tokens_) {
      const size_t found = text.find(special.first, position);
      if (found != std::string::npos && found < next) next = found;
    }
    for (const std::string& segment :
         SplitSegments(text.substr(position, next - position))) {
      EncodeSegment(segment, &output);
    }
    position = next;
  }
  return output;
}

std::string GgufTokenizer::DecodePiece(const std::string& piece) const {
  std::string bytes;
  for (const std::string& character : SplitUtf8(piece)) {
    const auto found = char_to_byte_.find(character);
    if (found != char_to_byte_.end()) {
      bytes += static_cast<char>(found->second);
    } else {
      bytes += character;
    }
  }
  return bytes;
}

std::string GgufTokenizer::DecodeToken(int token) const {
  if (token < 0 || token >= static_cast<int>(tokens_.size())) {
    return "";
  }
  if (token_bytes_[static_cast<size_t>(token)] >= 0) {
    return std::string(1, static_cast<char>(token_bytes_[token]));
  }
  return DecodePiece(tokens_[static_cast<size_t>(token)]);
}

std::string GgufTokenizer::Decode(const std::vector<int>& tokens) const {
  std::string text;
  for (int token : tokens) {
    text += DecodeToken(token);
  }
  return text;
}

std::unique_ptr<ITokenizer> CreateGgufTokenizer(const std::string& path) {
  auto tokenizer = std::make_unique<GgufTokenizer>();
  if (!tokenizer->Load(path)) return nullptr;
  return tokenizer;
}

}  // namespace mini_lla
