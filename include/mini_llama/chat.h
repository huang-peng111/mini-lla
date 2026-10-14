#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_CHAT_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_CHAT_H_

#include <string>
#include <vector>

namespace mini_lla {

// 一条聊天消息，role 取 system、user 或 assistant。
struct ChatMessage {
  std::string role;
  std::string content;
};

// 保存多轮消息，并把消息转换为模型可读的 prompt。
class ChatSession {
 public:
  // 创建空会话。
  ChatSession() = default;
  // 追加一条消息。
  void add_message(std::string role, std::string content);
  // 清空所有历史消息。
  void clear();
  // 返回当前消息数量。
  size_t size() const { return messages_.size(); }
  // 使用简单角色标签构造完整 prompt。
  std::string build_prompt() const;

 private:
  std::vector<ChatMessage> messages_;
};

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_CHAT_H_
