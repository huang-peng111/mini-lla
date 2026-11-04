#include "mini_llama/cuda_runtime.h"
#include "test_main.h"

using namespace mini_lla;

// CPU-only 构建下 CUDA 接口应给出明确错误;启用 CUDA 时检查设备信息。
void TestCudaRuntime() {
  if (!CudaRuntimeBuilt()) {
    CHECK_THROWS(CudaDeviceCount());
    DeviceBuffer buffer;
    CHECK_THROWS(buffer.Reset(1024));
    CHECK_TRUE(buffer.empty());
    return;
  }

  const int devices = CudaDeviceCount();
  CHECK_TRUE(devices >= 0);
  if (devices == 0) return;
  const CudaDeviceInfo info = CudaGetDeviceInfo(0);
  CHECK_TRUE(!info.name.empty());
  CHECK_TRUE(info.total_memory > 0);
  CHECK_TRUE(!CudaFormatDeviceInfo(info).empty());
}
