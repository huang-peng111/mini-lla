#include "mini_llama/cuda_ops.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "mini_llama/cuda_runtime.h"
#include "mini_llama/quantized_tensor.h"

namespace mini_lla {
namespace {

// 把 FP16 位模式还原成 F32。
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

// 一个 block 完成一次 RMSNorm:x 与 weight 长度都是 size。
__global__ void RmsNormKernel(const float* x, const float* weight, float* y,
                              int size, float epsilon) {
  __shared__ float shared[256];
  float sum = 0.0f;
  for (int i = threadIdx.x; i < size; i += blockDim.x) sum += x[i] * x[i];
  shared[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float scale = rsqrtf(shared[0] / static_cast<float>(size) + epsilon);
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    y[i] = x[i] * scale * weight[i];
  }
}

// 每个线程处理一对通道,完成一个 head 中的旋转。
__global__ void RopeKernel(float* data, int heads, int head_dim, int position,
                           float theta, int neox) {
  const int half = head_dim / 2;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= heads * half) return;

  const int head = index / half;
  const int i = index % half;
  const float exponent = 2.0f * static_cast<float>(i) /
                         static_cast<float>(head_dim);
  const float angle =
      static_cast<float>(position) / powf(theta, exponent);
  const float cosine = cosf(angle);
  const float sine = sinf(angle);

  const int first = head * head_dim + (neox != 0 ? i : 2 * i);
  const int second = head * head_dim + (neox != 0 ? i + half : 2 * i + 1);
  const float a = data[first];
  const float b = data[second];
  data[first] = a * cosine - b * sine;
  data[second] = a * sine + b * cosine;
}

// 逐元素 SiLU。
__global__ void SiluKernel(const float* x, float* y, int size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) y[i] = x[i] / (1.0f + expf(-x[i]));
}

// 逐元素乘法。
__global__ void MulKernel(const float* a, const float* b, float* y, int size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) y[i] = a[i] * b[i];
}

// 逐元素加法,可选地加上偏置。
__global__ void AddKernel(const float* a, const float* b, float* y, int size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) y[i] = a[i] + b[i];
}

// 拷贝并叠加偏置,bias 为空时只拷贝。
__global__ void AddBiasKernel(const float* value, const float* bias, float* y,
                              int size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) y[i] = value[i] + (bias == nullptr ? 0.0f : bias[i]);
}

// 一个 block 完成一维 softmax。
__global__ void SoftmaxKernel(const float* x, float* y, int size) {
  __shared__ float shared[256];
  float maximum = -INFINITY;
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    maximum = fmaxf(maximum, x[i]);
  }
  shared[threadIdx.x] = maximum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] = fmaxf(shared[threadIdx.x],
                                  shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  maximum = shared[0];

  float total = 0.0f;
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    total += expf(x[i] - maximum);
  }
  shared[threadIdx.x] = total;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  total = shared[0];
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    y[i] = expf(x[i] - maximum) / total;
  }
}

// 从 F32 embedding 中取出第 token 行。
__global__ void EmbeddingF32Kernel(const float* weight, float* out, int token,
                                   int cols) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < cols) out[i] = weight[static_cast<size_t>(token) * cols + i];
}

// 从 Q8_0 embedding 中取出第 token 行。
__global__ void EmbeddingQ8Kernel(const BlockQ8_0* weight, float* out,
                                  int token, int cols, int blocks_per_row) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cols) return;
  const BlockQ8_0& block =
      weight[static_cast<size_t>(token) * blocks_per_row + i / kBlockSize];
  out[i] = Fp16ToFloat(block.d) * static_cast<float>(block.qs[i % kBlockSize]);
}

// 从 Q4_0 embedding 中取出第 token 行。
__global__ void EmbeddingQ40Kernel(const BlockQ4_0* weight, float* out,
                                   int token, int cols, int blocks_per_row) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cols) return;
  const BlockQ4_0& block =
      weight[static_cast<size_t>(token) * blocks_per_row + i / kBlockSize];
  const int index = i % kBlockSize;
  const int packed = index < kBlockSize / 2
                         ? block.qs[index] & 0x0f
                         : block.qs[index - kBlockSize / 2] >> 4;
  out[i] = Fp16ToFloat(block.d) * static_cast<float>(packed - 8);
}

// 从 Q4_1 embedding 中取出第 token 行。
__global__ void EmbeddingQ41Kernel(const BlockQ4_1* weight, float* out,
                                   int token, int cols, int blocks_per_row) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cols) return;
  const BlockQ4_1& block =
      weight[static_cast<size_t>(token) * blocks_per_row + i / kBlockSize];
  const int index = i % kBlockSize;
  const int packed = index < kBlockSize / 2
                         ? block.qs[index] & 0x0f
                         : block.qs[index - kBlockSize / 2] >> 4;
  out[i] = Fp16ToFloat(block.d) * static_cast<float>(packed) +
           Fp16ToFloat(block.m);
}

}  // namespace

CudaTensor CudaRmsNorm(const CudaTensor& x, const CudaWeight& weight,
                       float epsilon) {
  if (weight.rows != static_cast<int>(x.Size())) {
    throw std::invalid_argument("CudaRmsNorm: weight size mismatch");
  }
  CudaTensor output(x.shape());
  RmsNormKernel<<<1, 256>>>(x.data(),
                            static_cast<const float*>(weight.buffer.data()),
                            output.data(), static_cast<int>(x.Size()), epsilon);
  CudaCheckLastError("CudaRmsNorm");
  return output;
}

void CudaRope(CudaTensor* q, CudaTensor* k, int position, float theta,
              int n_heads, int n_kv_heads, int head_dim, RopeType rope_type) {
  if (q == nullptr || k == nullptr) {
    throw std::invalid_argument("CudaRope: null tensor");
  }
  const int neox = rope_type == RopeType::kNeoX ? 1 : 0;
  const int threads = 128;
  const int q_pairs = n_heads * (head_dim / 2);
  const int k_pairs = n_kv_heads * (head_dim / 2);
  RopeKernel<<<(q_pairs + threads - 1) / threads, threads>>>(
      q->data(), n_heads, head_dim, position, theta, neox);
  RopeKernel<<<(k_pairs + threads - 1) / threads, threads>>>(
      k->data(), n_kv_heads, head_dim, position, theta, neox);
  CudaCheckLastError("CudaRope");
}

CudaTensor CudaSilu(const CudaTensor& x) {
  CudaTensor output(x.shape());
  const int size = static_cast<int>(x.Size());
  if (size == 0) return output;
  SiluKernel<<<(size + 255) / 256, 256>>>(x.data(), output.data(), size);
  CudaCheckLastError("CudaSilu");
  return output;
}

CudaTensor CudaMul(const CudaTensor& a, const CudaTensor& b) {
  if (a.Size() != b.Size()) throw std::invalid_argument("CudaMul: size mismatch");
  CudaTensor output(a.shape());
  const int size = static_cast<int>(a.Size());
  if (size == 0) return output;
  MulKernel<<<(size + 255) / 256, 256>>>(a.data(), b.data(), output.data(),
                                         size);
  CudaCheckLastError("CudaMul");
  return output;
}

CudaTensor CudaAdd(const CudaTensor& a, const CudaTensor& b) {
  if (a.Size() != b.Size()) throw std::invalid_argument("CudaAdd: size mismatch");
  CudaTensor output(a.shape());
  const int size = static_cast<int>(a.Size());
  if (size == 0) return output;
  AddKernel<<<(size + 255) / 256, 256>>>(a.data(), b.data(), output.data(),
                                         size);
  CudaCheckLastError("CudaAdd");
  return output;
}

CudaTensor CudaSoftmax(const CudaTensor& x) {
  if (x.shape().size() != 1 || x.Size() == 0) {
    throw std::invalid_argument("CudaSoftmax: expected a non-empty vector");
  }
  CudaTensor output(x.shape());
  SoftmaxKernel<<<1, 256>>>(x.data(), output.data(),
                            static_cast<int>(x.Size()));
  CudaCheckLastError("CudaSoftmax");
  return output;
}

CudaTensor CudaEmbeddingRow(const CudaWeight& embedding, int token) {
  if (token < 0 || token >= embedding.rows) {
    throw std::out_of_range("CudaEmbeddingRow: token out of range");
  }
  CudaTensor output({embedding.cols});
  const int cols = embedding.cols;
  const int blocks_per_row = (cols + kBlockSize - 1) / kBlockSize;
  const int blocks = (cols + 255) / 256;
  switch (embedding.type) {
    case QuantType::kF32:
      EmbeddingF32Kernel<<<blocks, 256>>>(
          static_cast<const float*>(embedding.buffer.data()), output.data(),
          token, cols);
      break;
    case QuantType::kQ8_0:
      EmbeddingQ8Kernel<<<blocks, 256>>>(
          static_cast<const BlockQ8_0*>(embedding.buffer.data()), output.data(),
          token, cols, blocks_per_row);
      break;
    case QuantType::kQ4_0:
      EmbeddingQ40Kernel<<<blocks, 256>>>(
          static_cast<const BlockQ4_0*>(embedding.buffer.data()),
          output.data(), token, cols, blocks_per_row);
      break;
    case QuantType::kQ4_1:
      EmbeddingQ41Kernel<<<blocks, 256>>>(
          static_cast<const BlockQ4_1*>(embedding.buffer.data()),
          output.data(), token, cols, blocks_per_row);
      break;
  }
  CudaCheckLastError("CudaEmbeddingRow");
  return output;
}

CudaTensor CudaAddBias(const CudaTensor& value, const CudaWeight* bias) {
  CudaTensor output(value.shape());
  const int size = static_cast<int>(value.Size());
  if (size == 0) return output;
  const float* bias_data =
      bias == nullptr ? nullptr : static_cast<const float*>(bias->buffer.data());
  AddBiasKernel<<<(size + 255) / 256, 256>>>(value.data(), bias_data,
                                             output.data(), size);
  CudaCheckLastError("CudaAddBias");
  return output;
}

}  // namespace mini_lla
