#ifndef MINI_LLA_INCLUDE_MINI_LLAMA_OPS_H_
#define MINI_LLA_INCLUDE_MINI_LLAMA_OPS_H_

#include "mini_llama/tensor.h"
namespace mini_lla {

// CPU 基础算子，供后续 Transformer 前向计算复用。
Tensor matmul(const Tensor& a,const Tensor& b);
Tensor linear(const Tensor& x,const Tensor& weight,const Tensor* bias=nullptr);
Tensor rms_norm(const Tensor& x,const Tensor& weight,float eps=1e-5f);
Tensor softmax(const Tensor& x);
Tensor silu(const Tensor& x);
Tensor swiglu(const Tensor& gate,const Tensor& up);
void rope(Tensor& q,Tensor& k,int position,float theta=10000.0f);
Tensor add(const Tensor& a,const Tensor& b);
int argmax(const Tensor& x);
}

#endif  // MINI_LLA_INCLUDE_MINI_LLAMA_OPS_H_
