#include <memory>
#include <string>
#include <vector>

#include "gguf_fixture.h"
#include "mini_llama/gguf_tokenizer.h"
#include "mini_llama/tokenizer.h"
#include "test_main.h"

using namespace mini_lla;

// 覆盖 GGUF 字节级 BPE 的编码、解码、特殊 token 和字节回退。
void TestTokenizer() {
  const std::string path = test::TempPath("mini_lla_tokenizer.gguf");
  test::WriteFixtureGguf(path);

  const std::unique_ptr<ITokenizer> tokenizer = CreateGgufTokenizer(path);
  CHECK_TRUE(tokenizer != nullptr);
  CHECK_EQUAL(tokenizer->vocab_size(), 10);
  CHECK_EQUAL(tokenizer->bos_id(), 0);
  CHECK_EQUAL(tokenizer->eos_id(), 2);
  CHECK_EQUAL(tokenizer->unk_id(), 0);

  // 字节级 BPE:先合并 "h" 和 "i",再查词表得到 id 5。
  const std::vector<int> merged = tokenizer->Encode("hi");
  CHECK_EQUAL(merged.size(), 1u);
  CHECK_EQUAL(merged[0], 5);

  // 空格和单词一起切分," h" 合并成 "Ġh"。
  const std::vector<int> spaced = tokenizer->Encode(" h");
  CHECK_EQUAL(spaced.size(), 1u);
  CHECK_EQUAL(spaced[0], 7);

  // 合并顺序按 rank 从小到大," hi" 先合并 "hi"。
  const std::vector<int> ordered = tokenizer->Encode(" hi");
  CHECK_EQUAL(ordered.size(), 2u);
  CHECK_EQUAL(ordered[0], 6);
  CHECK_EQUAL(ordered[1], 5);

  // 标点单独成 token。
  const std::vector<int> punctuated = tokenizer->Encode("hi!");
  CHECK_EQUAL(punctuated.size(), 2u);
  CHECK_EQUAL(punctuated[0], 5);
  CHECK_EQUAL(punctuated[1], 8);

  // 特殊 token 整体匹配,不会被拆成普通字符。
  const std::vector<int> special =
      tokenizer->Encode("<|im_start|>hi<|im_end|>");
  CHECK_EQUAL(special.size(), 3u);
  CHECK_EQUAL(special[0], 1);
  CHECK_EQUAL(special[1], 5);
  CHECK_EQUAL(special[2], 2);

  // 词表里没有的字符退回字节 token;没有字节 token 时落到 UNK。
  const std::vector<int> byte_fallback = tokenizer->Encode("z");
  CHECK_EQUAL(byte_fallback.size(), 1u);
  CHECK_EQUAL(byte_fallback[0], 9);
  const std::vector<int> unknown = tokenizer->Encode("q");
  CHECK_EQUAL(unknown.size(), 1u);
  CHECK_EQUAL(unknown[0], 0);

  // 解码:普通 token 还原文本,字节 token 还原原始字节。
  CHECK_TRUE(tokenizer->Decode({3, 4}) == "hi");
  CHECK_TRUE(tokenizer->DecodeToken(5) == "hi");
  CHECK_TRUE(tokenizer->DecodeToken(9) == "z");
  CHECK_TRUE(tokenizer->DecodeToken(999).empty());

  // 空文本和 ASCII 路径。
  CHECK_TRUE(tokenizer->Encode("").empty());
  const std::unique_ptr<ITokenizer> ascii = CreateAsciiTokenizer();
  CHECK_EQUAL(ascii->vocab_size(), 128);
  CHECK_TRUE(ascii->Decode({72, 105}) == "Hi");
  CHECK_TRUE(CreateGgufTokenizer("no-such-file.gguf") == nullptr);
}
