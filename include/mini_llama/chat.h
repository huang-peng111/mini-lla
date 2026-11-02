#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_CHAT_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_CHAT_H_

#include <cstddef>
#include <string>
#include <vector>

namespace mini_lla {

// 一条聊天消息,role 取 system、user 或 assistant。
struct ChatMessage {
  std::string role;
  std::string content;
};

// 保存多轮对话的消息,供 prompt 构造使用。
class ChatSession {
 public:
  // 追加一条消息。
  void AddMessage(std::string role, std::string content);

  // 清空全部历史。
  void Clear();

  // 返回消息条数。
  size_t Size() const { return messages_.size(); }

  // 返回消息列表。
  const std::vector<ChatMessage>& Messages() const { return messages_; }

 private:
  std::vector<ChatMessage> messages_;
};

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_CHAT_H_
