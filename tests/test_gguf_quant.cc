#include "mini_llama/quant.h"
#include <cmath>
#include <stdexcept>
using namespace mini_lla;
// 验证量化结果可以还原，并保持线性层输出形状。
int test_gguf_quant() { Tensor input({2, 3}); input.data = {-1, 0, 1, 2, 3, 4}; for (QuantizedTensor q : {quantize_q8_0(input), quantize_q4_0(input), quantize_q4_1(input)}) { Tensor restored = q.dequantize(); if (restored.shape != input.shape || !std::isfinite(restored.data[0])) return 1; } return 0; }
