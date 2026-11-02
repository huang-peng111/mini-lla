#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_TOKENIZER_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_TOKENIZER_H_

#include <memory>
#include <string>
#include <vector>

namespace mini_lla {

// tokenizer 统一接口:负责文本与 token id 的相互转换。
class ITokenizer {
 public:
  virtual ~ITokenizer() = default;

  // 把文本编码为 token id 序列。
  virtual std::vector<int> Encode(const std::string& text) const = 0;

  // 把 token id 序列解码为文本。
  virtual std::string Decode(const std::vector<int>& tokens) const = 0;

  // 解码单个 token id,无效 id 返回空串。
  virtual std::string DecodeToken(int token) const = 0;

  // 返回词表大小。
  virtual int vocab_size() const = 0;

  // 返回起始 token id,-1 表示该模型不使用。
  virtual int bos_id() const = 0;

  // 返回结束 token id,-1 表示该模型不使用。
  virtual int eos_id() const = 0;

  // 返回未知 token id,-1 表示该模型没有未知 token。
  virtual int unk_id() const = 0;
};

// 教学版 ASCII tokenizer:一个可打印字符对应一个 token id。
class AsciiTokenizer : public ITokenizer {
 public:
  // 编码时在开头加入 BOS,非 ASCII 字符落到 UNK。
  std::vector<int> Encode(const std::string& text) const override;

  // 逐字符拼回文本,控制 token 不输出。
  std::string Decode(const std::vector<int>& tokens) const override;

  // 解码单个字符 token。
  std::string DecodeToken(int token) const override;

  int vocab_size() const override { return 128; }
  int bos_id() const override { return 1; }
  int eos_id() const override { return 2; }
  int unk_id() const override { return 0; }
};

// 创建 ASCII tokenizer。
std::unique_ptr<ITokenizer> CreateAsciiTokenizer();

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_TOKENIZER_H_
