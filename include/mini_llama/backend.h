#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_BACKEND_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_BACKEND_H_

#include <string>

namespace mini_lla {

// 推理后端类型。
enum class BackendKind { kCpu, kCuda };

// 后端配置:后端类型与设备编号。
struct BackendConfig {
  BackendKind kind = BackendKind::kCpu;
  int device = 0;
  bool device_set = false;
};

// 解析 "cpu" 或 "cuda",失败返回 false。
bool ParseBackend(const std::string& text, BackendKind* kind);

// 返回后端的名字。
const char* BackendName(BackendKind kind);

// 校验后端在当前构建与当前机器上是否可用,不可用时抛出异常。
void ValidateBackend(const BackendConfig& config);

// 返回一行执行路径说明,供命令行输出。
std::string BackendNote(const BackendConfig& config);

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_BACKEND_H_
