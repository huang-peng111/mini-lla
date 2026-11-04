#include "mini_llama/cuda_matmul.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>

#include "mini_llama/cuda_runtime.h"
#include "mini_llama/quantized_tensor.h"

namespace mini_lla {
namespace {

// 把 FP16 位模式还原成 F32,用于量化 block 的系数。
__device__ float Fp16ToFloat(unsigned short value) {
  const unsigned int sign = static_cast<unsigned int>(value & 0x8000u) << 16;
  unsigned int exponent = (value >> 10) & 0x1fu;
  unsigned int mantissa = value & 0x03ffu;
  unsigned int bits = 0;
  if (exponent == 0) {
    if (mantissa != 0) {
      exponent = 1;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03ffu;
      bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    } else {
      bits = sign;
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
  }
  return __uint_as_float(bits);
}

// 每个线程计算一个输出元素,y[row] = dot(x, weight[row])。
__global__ void GemvQ8Kernel(const float* x, const BlockQ8_0* weight, float* y,
                             int rows, int cols, int blocks_per_row) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  float sum = 0.0f;
  for (int b = 0; b < blocks_per_row; ++b) {
    const BlockQ8_0& block = weight[row * blocks_per_row + b];
    const float scale = Fp16ToFloat(block.d);
    const int begin = b * kBlockSize;
    const int end = begin + kBlockSize < cols ? begin + kBlockSize : cols;
    for (int i = begin; i < end; ++i) {
      sum += scale * static_cast<float>(block.qs[i - begin]) * x[i];
    }
  }
  y[row] = sum;
}

// Q4_0 版本:前半元素在低 4 位,后半元素在高 4 位,都减去 8。
__global__ void GemvQ40Kernel(const float* x, const BlockQ4_0* weight, float* y,
                              int rows, int cols, int blocks_per_row) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  float sum = 0.0f;
  for (int b = 0; b < blocks_per_row; ++b) {
    const BlockQ4_0& block = weight[row * blocks_per_row + b];
    const float scale = Fp16ToFloat(block.d);
    const int begin = b * kBlockSize;
    const int end = begin + kBlockSize < cols ? begin + kBlockSize : cols;
    for (int i = begin; i < end; ++i) {
      const int index = i - begin;
      const int packed = index < kBlockSize / 2
                             ? block.qs[index] & 0x0f
                             : block.qs[index - kBlockSize / 2] >> 4;
      sum += scale * static_cast<float>(packed - 8) * x[i];
    }
  }
  y[row] = sum;
}

// Q4_1 版本:量化值在 [0, 15],反量化后再加最小值。
__global__ void GemvQ41Kernel(const float* x, const BlockQ4_1* weight, float* y,
                              int rows, int cols, int blocks_per_row) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  float sum = 0.0f;
  for (int b = 0; b < blocks_per_row; ++b) {
    const BlockQ4_1& block = weight[row * blocks_per_row + b];
    const float scale = Fp16ToFloat(block.d);
    const float minimum = Fp16ToFloat(block.m);
    const int begin = b * kBlockSize;
    const int end = begin + kBlockSize < cols ? begin + kBlockSize : cols;
    for (int i = begin; i < end; ++i) {
      const int index = i - begin;
      const int packed = index < kBlockSize / 2
                             ? block.qs[index] & 0x0f
                             : block.qs[index - kBlockSize / 2] >> 4;
      sum += (scale * static_cast<float>(packed) + minimum) * x[i];
    }
  }
  y[row] = sum;
}

// 每个设备复用一个 cuBLAS 句柄;本项目只支持单设备。
cublasHandle_t& CublasHandle() {
  static cublasHandle_t handle = nullptr;
  if (handle == nullptr) {
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("创建 cuBLAS 句柄失败");
    }
  }
  return handle;
}

}  // namespace

CudaTensor CudaLinear(const CudaTensor& x, const CudaWeight& weight) {
  if (x.Size() != static_cast<size_t>(weight.cols)) {
    throw std::invalid_argument("CudaLinear: input size mismatch");
  }
  CudaTensor output({weight.rows});

  if (weight.type == QuantType::kF32) {
    // 权重按 [rows, cols] 行主序存放,按列主序解释就是 [cols, rows] 的转置。
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t status = cublasSgemv(
        CublasHandle(), CUBLAS_OP_T, weight.cols, weight.rows, &alpha,
        static_cast<const float*>(weight.buffer.data()), weight.cols,
        x.data(), 1, &beta, output.data(), 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("cuBLAS SGEMV 执行失败");
    }
    return output;
  }

  const int blocks_per_row = (weight.cols + kBlockSize - 1) / kBlockSize;
  const int threads = 256;
  const int blocks = (weight.rows + threads - 1) / threads;
  switch (weight.type) {
    case QuantType::kQ8_0:
      GemvQ8Kernel<<<blocks, threads>>>(
          x.data(), static_cast<const BlockQ8_0*>(weight.buffer.data()),
          output.data(), weight.rows, weight.cols, blocks_per_row);
      break;
    case QuantType::kQ4_0:
      GemvQ40Kernel<<<blocks, threads>>>(
          x.data(), static_cast<const BlockQ4_0*>(weight.buffer.data()),
          output.data(), weight.rows, weight.cols, blocks_per_row);
      break;
    case QuantType::kQ4_1:
      GemvQ41Kernel<<<blocks, threads>>>(
          x.data(), static_cast<const BlockQ4_1*>(weight.buffer.data()),
          output.data(), weight.rows, weight.cols, blocks_per_row);
      break;
    case QuantType::kF32:
      break;
  }
  CudaCheckLastError("CudaLinear");
  return output;
}

}  // namespace mini_lla
