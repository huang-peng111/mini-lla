#include "mini_llama/prompt_builder.h"

namespace mini_lla {
namespace {

// 通用模板:System/User/Assistant 文本标签,最后一轮等待 assistant 回复。
std::string BuildPlain(const std::vector<ChatMessage>& messages) {
  std::string prompt;
  for (const ChatMessage& message : messages) {
    if (message.role == "system") {
      prompt += "System: " + message.content + "\n";
    } else if (message.role == "user") {
      prompt += "User: " + message.content + "\n";
    } else if (message.role == "assistant") {
      prompt += "Assistant: " + message.content + "\n";
    }
  }
  prompt += "Assistant:";
  return prompt;
}

// Qwen2 模板:没有 system 消息时补一条默认的,末尾留出 assistant 回合。
std::string BuildQwen2(const std::vector<ChatMessage>& messages) {
  std::string prompt;
  bool has_system = false;
  for (const ChatMessage& message : messages) {
    if (message.role == "system") has_system = true;
  }
  if (!has_system) {
    prompt += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
  }
  for (const ChatMessage& message : messages) {
    if (message.role != "system" && message.role != "user" &&
        message.role != "assistant") {
      continue;
    }
    prompt += "<|im_start|>" + message.role + "\n" + message.content +
              "<|im_end|>\n";
  }
  prompt += "<|im_start|>assistant\n";
  return prompt;
}

}  // namespace

std::string BuildPrompt(const std::vector<ChatMessage>& messages,
                        const std::string& template_name) {
  if (template_name == kQwen2Template) return BuildQwen2(messages);
  return BuildPlain(messages);
}

}  // namespace mini_lla
