# mini-lla：Tensor 与基础算子

这是从零复刻 mini-llama.cpp 的第一个独立阶段，对应“Tensor 与基础算子”章节。当前只实现 CPU F32 数据结构和 Transformer 前向所需的数学算子，暂不包含模型加载、tokenizer、KV cache、GGUF、量化或 CUDA。

## 构建与测试（Windows PowerShell）

```powershell
cmake -S . -B build
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

核心代码：`include/mini_llama/tensor.h`、`src/tensor.cc`、`include/mini_llama/ops.h`、`src/ops.cc`。测试覆盖 Tensor 创建/索引/reshape、矩阵乘、RMSNorm、Softmax、RoPE、激活函数和 argmax。

## 当前接口

- `Tensor`：连续 F32 内存、shape、索引、reshape。
- `matmul` / `linear`：二维矩阵乘和线性层。
- `rms_norm`、`softmax`、`silu`、`swiglu`。
- `rope`：对 Q/K 的偶数维执行旋转位置编码。
- `add`、`argmax`。

下一阶段会在此基础上增加 Tiny 模型配置和权重加载。

