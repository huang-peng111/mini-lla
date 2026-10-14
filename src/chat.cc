#include "mini_llama/chat.h"

#include <utility>

namespace mini_lla {

void ChatSession::add_message(std::string role, std::string content) {
  messages_.push_back({std::move(role), std::move(content)});
}

void ChatSession::clear() { messages_.clear(); }

std::string ChatSession::build_prompt() const {
  std::string prompt;
  for (const ChatMessage& message : messages_) {
    prompt += "<|" + message.role + "|>\n";
    prompt += message.content;
    prompt += "\n";
  }
  prompt += "<|assistant|>\n";
  return prompt;
}

}  // namespace mini_lla
