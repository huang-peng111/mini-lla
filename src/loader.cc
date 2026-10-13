#include <fstream>
#include <regex>
#include <stdexcept>

#include "mini_llama/model.h"
namespace mini_lla {
namespace {
// 读取 manifest 的全部文本内容。
std::string read_text(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open manifest: " + path);
  return {std::istreambuf_iterator<char>(input), {}};
}
// 从 manifest 中读取整数配置，找不到时使用默认值。
int number(const std::string& text, const std::string& name, int fallback) {
  std::smatch match;
  std::regex pattern("\\\"" + name + "\\\"\\s*:\\s*(-?[0-9]+)");
  return std::regex_search(text, match, pattern) ? std::stoi(match[1])
                                                 : fallback;
}
// 从二进制权重文件读取一个连续 F32 张量。
Tensor read_tensor(std::ifstream& input, size_t offset,
                   const std::vector<int>& shape) {
  // manifest 中的 offset 指向 model.bin 内的 F32 连续数据。
  Tensor tensor(shape);
  input.seekg(static_cast<std::streamoff>(offset));
  input.read(reinterpret_cast<char*>(tensor.data.data()),
             static_cast<std::streamsize>(tensor.data.size() * sizeof(float)));
  if (!input) throw std::runtime_error("model.bin is truncated");
  return tensor;
}
}  // namespace
Model load_model(const std::string& manifest_path,
                 const std::string& weights_path) {
  const std::string text = read_text(manifest_path);
  Model model;
  auto& c = model.config;
  c.vocab_size = number(text, "vocab_size", c.vocab_size);
  c.dim = number(text, "dim", c.dim);
  c.hidden_dim = number(text, "hidden_dim", c.hidden_dim);
  c.n_layers = number(text, "n_layers", c.n_layers);
  c.n_heads = number(text, "n_heads", c.n_heads);
  c.n_kv_heads = number(text, "n_kv_heads", c.n_heads);
  c.max_seq_len = number(text, "max_seq_len", c.max_seq_len);
  std::ifstream input(weights_path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open weights: " + weights_path);
  // 按导出脚本规定的固定顺序读取权重，保持 loader 简单可读。
  size_t offset = 0;
  auto load = [&](std::vector<int> shape) {
    Tensor result = read_tensor(input, offset, shape);
    offset += result.size() * sizeof(float);
    return result;
  };
  model.token_embedding = load({c.vocab_size, c.dim});
  model.layers.resize(c.n_layers);
  for (auto& layer : model.layers) {
    layer.attention_norm = load({c.dim});
    layer.wq = load({c.dim, c.dim});
    layer.wk = load({c.dim, c.dim});
    layer.wv = load({c.dim, c.dim});
    layer.wo = load({c.dim, c.dim});
    layer.ffn_norm = load({c.dim});
    layer.w_gate = load({c.hidden_dim, c.dim});
    layer.w_up = load({c.hidden_dim, c.dim});
    layer.w_down = load({c.dim, c.hidden_dim});
  }
  model.final_norm = load({c.dim});
  model.lm_head = load({c.vocab_size, c.dim});
  return model;
}
}  // namespace mini_lla
