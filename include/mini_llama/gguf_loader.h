#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_GGUF_LOADER_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_GGUF_LOADER_H_

#include <string>

#include "mini_llama/model.h"

namespace mini_lla {

// 加载真实模型时允许的最大上下文长度。
// 元数据里常见几万长度,按上限裁剪可以限制 KV cache 的内存占用。
constexpr int kMaxContextLength = 2048;

// 从 GGUF 文件加载模型权重,失败时抛出 std::runtime_error。
Model LoadGgufModel(const std::string& path);

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_GGUF_LOADER_H_
