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

前面的阶段已经完成 Tiny 模型配置、权重加载和基础推理。

## 当前命令

```powershell
# Greedy 或温度/top-k 采样生成
.\build\Debug\mini-lla.exe generate --model models\tiny --prompt "hello" --n 8
.\build\Debug\mini-lla.exe generate --model models\tiny --prompt "hello" --n 8 --temperature 0.8 --top-k 8 --seed 42

# 多轮聊天；输入 /help、/clear 或 /exit
.\build\Debug\mini-lla.exe run --model models\tiny --temperature 0.8 --top-k 8 --seed 42
```

## 阶段 8–10

- 阶段 8：README 记录构建、运行、测试和已知限制。
- 阶段 9：`Sampler` 支持 greedy、temperature、top-k 和固定 seed。
- 阶段 10：`ChatSession` 支持 user/assistant 消息、`/help`、`/clear`、`/exit`。

Tiny 模型使用随机权重，只用于验证推理链路，所以生成内容没有真实对话质量。

