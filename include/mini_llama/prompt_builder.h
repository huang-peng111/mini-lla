#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_PROMPT_BUILDER_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_PROMPT_BUILDER_H_

#include <string>
#include <vector>

#include "mini_llama/chat.h"

namespace mini_lla {

// 内置聊天模板标识:qwen2 使用 <|im_start|> 标签,空串使用通用标签。
constexpr char kQwen2Template[] = "qwen2";

// 把消息列表拼成模型输入的 prompt。
std::string BuildPrompt(const std::vector<ChatMessage>& messages,
                        const std::string& template_name);

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_PROMPT_BUILDER_H_
