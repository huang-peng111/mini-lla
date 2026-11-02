#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_GGUF_TOKENIZER_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_GGUF_TOKENIZER_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mini_llama/tokenizer.h"

namespace mini_lla {

// 从 GGUF metadata 构造的字节级 BPE tokenizer(GPT-2 风格)。
// 读取 tokenizer.ggml.tokens、merges、token_type 和 BOS/EOS/UNK 配置。
class GgufTokenizer : public ITokenizer {
 public:
  GgufTokenizer() = default;

  // 从 GGUF 文件加载词表;缺少必要字段时返回 false。
  bool Load(const std::string& path);

  // 编码文本:先匹配特殊 token,其余部分做字节级 BPE。
  std::vector<int> Encode(const std::string& text) const override;

  // 解码 token 序列,字节 token 会还原成原始字节。
  std::string Decode(const std::vector<int>& tokens) const override;

  // 解码单个 token id。
  std::string DecodeToken(int token) const override;

  int vocab_size() const override { return static_cast<int>(tokens_.size()); }
  int bos_id() const override { return bos_id_; }
  int eos_id() const override { return eos_id_; }
  int unk_id() const override { return unk_id_; }

 private:
  // 建立 GPT-2 风格的字节 <-> unicode 映射。
  void BuildByteMap();

  // 读取词表、merges 和 token 类型。
  bool LoadVocabulary(const std::string& path);

  // 根据 token 类型表分出特殊 token 和字节 token。
  void ClassifyTokens(const std::vector<int32_t>& types);

  // 对单个片段做字节级 BPE 并追加 token id。
  void EncodeSegment(const std::string& segment,
                     std::vector<int>* output) const;

  // 片段不在词表时,按字节退回字节 token 或 UNK。
  void AppendByteFallback(const std::string& piece,
                          std::vector<int>* output) const;

  // 把 token 文本中的 unicode 字符还原成原始字节。
  std::string DecodePiece(const std::string& piece) const;

  std::vector<std::string> tokens_;                 // id -> 文本
  std::unordered_map<std::string, int> token_ids_;  // 文本 -> id
  std::map<std::pair<std::string, std::string>, int> merge_ranks_;
  std::vector<std::pair<std::string, int>> special_tokens_;  // 长度降序
  std::vector<std::string> byte_to_char_;                    // 256 项
  std::unordered_map<std::string, uint8_t> char_to_byte_;
  std::vector<int> byte_token_ids_;  // 每个字节对应的 token id,-1 表示没有
  std::vector<int> token_bytes_;     // 每个 id 的字节值,-1 表示不是字节 token
  bool add_bos_ = false;
  int bos_id_ = -1;
  int eos_id_ = -1;
  int unk_id_ = -1;
};

// 创建 GGUF tokenizer,失败时返回 nullptr。
std::unique_ptr<ITokenizer> CreateGgufTokenizer(const std::string& path);

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_GGUF_TOKENIZER_H_
