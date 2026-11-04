#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_THREAD_POOL_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_THREAD_POOL_H_

#include <functional>

namespace mini_lla {

// 返回并行计算使用的线程数。
int ThreadCount();

// 设置并行计算的线程数,0 表示恢复成硬件线程数。
// 只能在没有任何并行任务运行时调用。
void SetThreadCount(int count);

// 把 [0, total) 切成若干块交给常驻线程池执行。
// 每个下标只会被处理一次;任务里抛出的异常会在调用线程重新抛出。
// 不可重入:不要在 body 内部再次调用 ParallelFor。
void ParallelFor(int total,
                 const std::function<void(int begin, int end)>& body);

}  // namespace mini_lla

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_THREAD_POOL_H_
