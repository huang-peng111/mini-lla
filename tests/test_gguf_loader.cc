#include <cmath>
#include <string>
#include <vector>

#include "gguf_fixture.h"
#include "mini_llama/batch.h"
#include "mini_llama/forward.h"
#include "mini_llama/gguf_loader.h"
#include "mini_llama/loader.h"
#include "mini_llama/prompt_builder.h"
#include "test_main.h"

using namespace mini_lla;

// 覆盖 GGUF 到模型的映射、量化权重类型和 GQA 前向。
void TestGgufLoader() {
  const std::string path = test::TempPath("mini_lla_model.gguf");
  test::WriteFixtureGguf(path);

  const Model model = LoadGgufModel(path);
  CHECK_EQUAL(model.config.vocab_size, 10);
  CHECK_EQUAL(model.config.dim, 32);
  CHECK_EQUAL(model.config.n_layers, 1);
  CHECK_EQUAL(model.config.n_heads, 2);
  CHECK_EQUAL(model.config.n_kv_heads, 1);
  CHECK_EQUAL(model.config.head_dim(), 16);
  CHECK_TRUE(model.config.rope_type == RopeType::kNeoX);
  CHECK_TRUE(model.tied_lm_head);

  // GGUF 里记录的量化类型被完整保留。
  CHECK_TRUE(model.token_embedding.type == QuantType::kQ8_0);
  CHECK_TRUE(model.layers[0].wq.type == QuantType::kQ8_0);
  CHECK_TRUE(model.layers[0].wv.type == QuantType::kQ4_0);
  CHECK_TRUE(model.layers[0].w_up.type == QuantType::kQ4_1);
  CHECK_EQUAL(model.layers[0].wq.rows, 32);
  CHECK_EQUAL(model.layers[0].wq.cols, 32);
  CHECK_EQUAL(model.layers[0].wk.rows, 16);
  CHECK_EQUAL(model.layers[0].wk.cols, 32);
  CHECK_EQUAL(model.layers[0].bq.Size(), 32u);
  CHECK_EQUAL(model.layers[0].bk.Size(), 16u);
  CHECK_EQUAL(model.layers[0].attention_norm.Size(), 32u);

  // GQA 前向:2 个 query head 共用 1 个 kv head。
  Context context(model.config);
  const MiniBatch batch = MiniBatch::FromTokens({1, 5, 7}, 0);
  const Tensor logits = ForwardBatch(context, model, batch);
  CHECK_EQUAL(logits.Size(), 10u);
  CHECK_EQUAL(context.position, 3);
  for (float value : logits.data) CHECK_TRUE(std::isfinite(value));

  // 同一输入重复推理结果一致。
  Context repeated(model.config);
  const Tensor again = ForwardBatch(repeated, model, batch);
  for (size_t i = 0; i < logits.Size(); ++i) {
    CHECK_NEAR(logits.data[i], again.data[i], 1e-6);
  }

  // 缺少必需的 tensor 时报错。
  const std::string incomplete = test::TempPath("mini_lla_incomplete.gguf");
  test::WriteFixtureGguf(incomplete, false);
  CHECK_THROWS(LoadGgufModel(incomplete));

  // 同一个文件也能创建 tokenizer 并识别聊天模板。
  const LoadedModel loaded = LoadModelAndTokenizer(path);
  CHECK_TRUE(loaded.tokenizer != nullptr);
  CHECK_EQUAL(loaded.tokenizer->vocab_size(), 10);
  CHECK_TRUE(loaded.chat_template == kQwen2Template);
}
