#include "mini_llama/kv_cache.h"
#include "test_main.h"

using namespace mini_lla;

// 覆盖 KV cache 的读写、层间隔离和越界检查。
void TestKvCache() {
  const ModelConfig config;
  Context context(config);
  CHECK_EQUAL(context.position, 0);

  Tensor key({config.n_kv_heads, config.head_dim()});
  Tensor value({config.n_kv_heads, config.head_dim()});
  for (size_t i = 0; i < key.Size(); ++i) {
    key.data[i] = static_cast<float>(i) + 1.0f;
    value.data[i] = static_cast<float>(i) + 100.0f;
  }

  context.cache.Write(0, 0, key, value);
  CHECK_NEAR(context.cache.KeyPtr(0, 0, 0)[1], 2.0f, 1e-6);
  CHECK_NEAR(context.cache.ValuePtr(0, 0, 0)[1], 101.0f, 1e-6);

  // 不同层互不影响。
  context.cache.Write(1, 0, key, value);
  key.data[0] = -1.0f;
  context.cache.Write(0, 1, key, value);
  CHECK_NEAR(context.cache.KeyPtr(0, 0, 0)[0], 1.0f, 1e-6);
  CHECK_NEAR(context.cache.KeyPtr(0, 1, 0)[0], -1.0f, 1e-6);
  CHECK_NEAR(context.cache.KeyPtr(1, 0, 0)[0], 1.0f, 1e-6);

  // 越界与形状不匹配都要报错。
  CHECK_THROWS(context.cache.Write(0, config.max_seq_len, key, value));
  CHECK_THROWS(context.cache.Write(config.n_layers, 0, key, value));
  CHECK_THROWS(context.cache.Write(0, 0, key, Tensor({config.head_dim()})));
  CHECK_THROWS(context.cache.KeyPtr(0, 0, config.n_kv_heads));
  CHECK_THROWS(context.cache.ValuePtr(0, -1, 0));
}
