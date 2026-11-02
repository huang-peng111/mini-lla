#include "mini_llama/batch.h"

#include <stdexcept>

#include "mini_llama/forward.h"

namespace mini_lla {

MiniBatch MiniBatch::Single(int token, int position) {
  MiniBatch batch;
  batch.tokens.push_back(token);
  batch.positions.push_back(position);
  return batch;
}

MiniBatch MiniBatch::FromTokens(const std::vector<int>& tokens, int start) {
  MiniBatch batch;
  batch.tokens = tokens;
  batch.positions.reserve(tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    batch.positions.push_back(start + static_cast<int>(i));
  }
  return batch;
}

Tensor ForwardBatch(Context& context, const Model& model,
                    const MiniBatch& batch) {
  if (batch.Size() == 0) {
    throw std::invalid_argument("ForwardBatch: empty batch");
  }
  if (batch.tokens.size() != batch.positions.size()) {
    throw std::invalid_argument("ForwardBatch: token/position size mismatch");
  }

  Tensor logits;
  for (int i = 0; i < batch.Size(); ++i) {
    context.position = batch.positions[static_cast<size_t>(i)];
    logits = ForwardToken(context, model, batch.tokens[static_cast<size_t>(i)]);
  }
  return logits;
}

}  // namespace mini_lla
