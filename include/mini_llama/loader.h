#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_LOADER_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_LOADER_H_

#include <memory>
#include <string>

#include "mini_llama/model.h"
#include "mini_llama/tokenizer.h"

namespace mini_lla {

// 一次加载的产物:模型权重、tokenizer 和聊天模板标识。
struct LoadedModel {
  Model model;
  std::shared_ptr<ITokenizer> tokenizer;
  std::string chat_template;  // "qwen2" 或空串(使用通用模板)
};

// 读取 tiny 目录中的 model.json 和 model.bin。
Model LoadTinyModel(const std::string& directory);

// 自动识别模型格式:目录按 tiny 格式加载,GGUF 文件按 GGUF 格式加载。
LoadedModel LoadModelAndTokenizer(const std::string& path);

// 从 GGUF 读取聊天模板标识;模板不受支持时返回空串。
std::string ChatTemplateFromGguf(const std::string& path);

// 打印模型配置和权重统计,供 inspect 命令使用。
void InspectModel(const Model& model, const std::string& source);

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_LOADER_H_
