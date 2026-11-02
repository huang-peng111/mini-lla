#include "mini_llama/chat.h"

#include <utility>

namespace mini_lla {

void ChatSession::AddMessage(std::string role, std::string content) {
  messages_.push_back({std::move(role), std::move(content)});
}

void ChatSession::Clear() { messages_.clear(); }

}  // namespace mini_lla
