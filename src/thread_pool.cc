#include "mini_llama/thread_pool.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace mini_lla {
namespace {

// 每块至少包含的元素个数,避免把很小的任务切得过碎。
constexpr int kMinChunk = 16;

// 返回硬件支持的线程数,查询失败时退回 4。
int HardwareThreads() {
  const unsigned int count = std::thread::hardware_concurrency();
  return count > 0 ? static_cast<int>(count) : 4;
}

// 常驻线程池:工作线程等待轮次号变化,然后用原子计数器领取分块。
class ThreadPool {
 public:
  // 返回全局唯一的线程池实例。
  static ThreadPool& Instance() {
    static ThreadPool pool;
    return pool;
  }

  // 设置工作线程数量,0 表示使用硬件线程数。
  void SetThreadCount(int count) {
    const int target = count > 0 ? count : HardwareThreads();
    if (target == GetThreadCount()) return;
    StopWorkers();
    StartWorkers(target);
  }

  // 返回当前工作线程数量。
  int GetThreadCount() {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(workers_.size());
  }

  // 执行一次并行任务,调用线程也会参与计算。
  void Run(int total, const std::function<void(int, int)>& body) {
    if (total <= 0) return;
    const int workers = GetThreadCount();
    if (workers <= 1 || total < workers * kMinChunk) {
      body(0, total);
      return;
    }

    std::unique_lock<std::mutex> lock(mutex_);
    const int chunk_size = (total + workers - 1) / workers;
    const int chunks = (total + chunk_size - 1) / chunk_size;
    if (chunks <= 1) {
      lock.unlock();
      body(0, total);
      return;
    }

    task_body_ = &body;
    task_total_ = total;
    task_chunk_size_ = chunk_size;
    task_chunks_ = chunks;
    next_chunk_ = 0;
    pending_ = static_cast<int>(workers_.size());
    error_ = nullptr;
    ++epoch_;
    start_cv_.notify_all();
    lock.unlock();

    RunChunks();

    lock.lock();
    done_cv_.wait(lock, [this] { return pending_ == 0; });
    const std::exception_ptr error = error_;
    task_body_ = nullptr;
    lock.unlock();
    if (error != nullptr) std::rethrow_exception(error);
  }

 private:
  // 创建线程池时按硬件线程数启动工作线程。
  ThreadPool() { StartWorkers(HardwareThreads()); }

  // 程序退出时唤醒并回收所有工作线程。
  ~ThreadPool() { StopWorkers(); }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  // 执行当前任务中尚未被领取的块,直到没有剩余。
  void RunChunks() {
    while (true) {
      const int chunk = next_chunk_.fetch_add(1);
      if (chunk >= task_chunks_) return;
      const int begin = chunk * task_chunk_size_;
      const int end = std::min(begin + task_chunk_size_, task_total_);
      try {
        (*task_body_)(begin, end);
      } catch (...) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        if (error_ == nullptr) error_ = std::current_exception();
      }
    }
  }

  // 工作线程主循环:等待新轮次,执行任务,最后通知调用线程。
  void WorkerLoop() {
    int seen_epoch = 0;
    while (true) {
      std::unique_lock<std::mutex> lock(mutex_);
      start_cv_.wait(lock, [this, seen_epoch] {
        return stop_ || epoch_ != seen_epoch;
      });
      if (stop_) return;
      seen_epoch = epoch_;
      lock.unlock();

      RunChunks();

      lock.lock();
      if (--pending_ == 0) done_cv_.notify_all();
    }
  }

  // 启动 count 个工作线程。
  void StartWorkers(int count) {
    std::lock_guard<std::mutex> lock(mutex_);
    workers_.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
      workers_.emplace_back([this] { WorkerLoop(); });
    }
  }

  // 通知所有工作线程退出并等待它们结束。
  void StopWorkers() {
    std::vector<std::thread> workers;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      workers.swap(workers_);
    }
    start_cv_.notify_all();
    for (std::thread& worker : workers) worker.join();
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = false;
  }

  std::mutex mutex_;
  std::mutex error_mutex_;
  std::condition_variable start_cv_;
  std::condition_variable done_cv_;
  std::vector<std::thread> workers_;

  // 当前任务的参数,由调用线程在持锁时设置。
  const std::function<void(int, int)>* task_body_ = nullptr;
  int task_total_ = 0;
  int task_chunk_size_ = 1;
  int task_chunks_ = 0;

  std::atomic<int> next_chunk_{0};
  int pending_ = 0;
  int epoch_ = 0;
  bool stop_ = false;
  std::exception_ptr error_;
};

}  // namespace

int ThreadCount() { return ThreadPool::Instance().GetThreadCount(); }

void SetThreadCount(int count) {
  ThreadPool::Instance().SetThreadCount(count);
}

void ParallelFor(int total,
                 const std::function<void(int begin, int end)>& body) {
  ThreadPool::Instance().Run(total, body);
}

}  // namespace mini_lla
