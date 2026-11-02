#include <cmath>

#include "mini_llama/batch.h"
#include "mini_llama/forward.h"
#include "mini_llama/loader.h"
#include "test_main.h"

using namespace mini_lla;

// 覆盖 tiny 模型的单步前向、批量等价性和错误路径。
void TestForward() {
  const Model model = LoadTinyModel(test::TinyModelDir());
  CHECK_EQUAL(model.config.dim, 32);
  CHECK_EQUAL(model.layers.size(), static_cast<size_t>(model.config.n_layers));

  Context context(model.config);
  const Tensor logits = ForwardToken(context, model, 3);
  CHECK_EQUAL(logits.Size(), static_cast<size_t>(model.config.vocab_size));
  CHECK_EQUAL(context.position, 1);
  for (float value : logits.data) CHECK_TRUE(std::isfinite(value));

  // 相同输入必须得到相同 logits。
  Context repeated(model.config);
  const Tensor again = ForwardToken(repeated, model, 3);
  for (size_t i = 0; i < logits.Size(); ++i) {
    CHECK_NEAR(logits.data[i], again.data[i], 1e-6);
  }

  // 批量 prefill 与逐 token 前向结果一致。
  const std::vector<int> tokens = {1, 5, 7};
  Context batched(model.config);
  const Tensor batch_logits =
      ForwardBatch(batched, model, MiniBatch::FromTokens(tokens, 0));
  Context stepwise(model.config);
  Tensor step_logits;
  for (int token : tokens) step_logits = ForwardToken(stepwise, model, token);
  CHECK_EQUAL(batched.position, stepwise.position);
  CHECK_EQUAL(batched.position, 3);
  for (size_t i = 0; i < batch_logits.Size(); ++i) {
    CHECK_NEAR(batch_logits.data[i], step_logits.data[i], 1e-5);
  }

  // 非法 token、越界位置和空 batch 都要报错。
  CHECK_THROWS(ForwardToken(context, model, -1));
  CHECK_THROWS(ForwardToken(context, model, model.config.vocab_size));
  context.position = model.config.max_seq_len;
  CHECK_THROWS(ForwardToken(context, model, 1));
  CHECK_THROWS(ForwardBatch(context, model, MiniBatch()));
}
