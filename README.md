# mini-lla

一个用 **C++17 从零实现**的 LLaMA 风格推理程序,目标是"可读、可测、可解释"。
不依赖 PyTorch、llama.cpp 或任何第三方推理库,自带最小测试框架与 CLI。

当前完成[实施路线](IMPLEMENTATION_ROADMAP.md)的阶段 0–21:CPU Tiny 闭环、
GGUF 真实模型、字节级 BPE、三种量化格式、CPU 多线程、benchmark,以及可选的
NVIDIA CUDA 后端。CUDA 路径的代码已完成,但**尚未在真实 GPU 上验证**(见下文)。

## 项目简介

- **模型格式**:Tiny(JSON manifest + F32 权重)、GGUF v2/v3。
- **架构**:LLaMA / Qwen2 命名,支持 GQA、RoPE(normal / NeoX)、注意力偏置、
  tied lm_head。
- **量化**:F32、Q8_0、Q4_0、Q4_1,block 布局与 ggml 一致,线性层逐 block
  即时反量化。
- **文本**:GGUF 内嵌词表的字节级 BPE,支持 merges、特殊 token 与字节回退;
  内置 Qwen2 与通用聊天模板。
- **执行**:CPU 多线程、可选 CUDA(F32 cuBLAS + 量化 GEMV + GPU 算子 + GPU KV cache)。
- **工程**:13 组单元测试 + 1 个 CLI 冒烟测试,`/W4` 零警告。

## 架构与数据流

```text
CLI(main.cc)
  ├─ loader.cc          目录 → Tiny;文件 → GGUF
  ├─ gguf_tokenizer.cc  字节级 BPE
  ├─ backend.cc         CPU / CUDA 选择与校验
  └─ forward.cc
       ├─ CPU:ops.cc + matmul_optimized.cc + quant.cc(线程池并行)
       └─ CUDA:cuda_forward.cc → cuda_matmul / cuda_ops / cuda_attention
  → sampler.cc → tokenizer.Decode → 终端输出
```

一次 `generate` 的路径:

```text
prompt → tokenizer.Encode → MiniBatch::FromTokens → ForwardBatch(prefill)
       → Sampler.Sample → ForwardToken(decode) → Decode → 增量打印
```

## 快速开始

CPU 构建(默认,不需要 CUDA Toolkit):

```powershell
cmake -S . -B build
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure

.\build\Debug\mini-lla.exe generate --model models\tiny -p "hello" -n 8
.\build\Debug\mini-lla.exe bench models\tiny -n 4 --threads 4
```

## 命令

| 命令 | 作用 |
|---|---|
| `generate` | 单次生成 |
| `run` | 多轮终端聊天(`/help` `/clear` `/exit`) |
| `bench` | prefill / decode 计时与吞吐 |
| `inspect` | 模型配置与权重统计 |
| `inspect-gguf` | GGUF metadata 与 tensor 表 |
| `--cuda-info` | CUDA 设备信息(CPU 构建会说明未启用) |

```powershell
# 真实 GGUF 模型(把路径换成你本机的文件)
.\build\Debug\mini-lla.exe generate model.gguf -p "Hello, my name is" -n 8 --ctx 256 --threads 4
.\build\Debug\mini-lla.exe run model.gguf -n 32 --ctx 512 --threads 4
.\build\Debug\mini-lla.exe bench model.gguf -p "Hello" -n 16 --threads 4 --verbose

# 固定种子的温度 / top-k 采样
.\build\Debug\mini-lla.exe generate --model models\tiny -n 8 --temperature 0.8 --top-k 8 --seed 42
```

| 选项 | 说明 |
|---|---|
| `--model <path>` | Tiny 目录或 GGUF 文件(也可写成位置参数) |
| `-p, --prompt <text>` | 输入文本,默认 `hello` |
| `-n, --n-predict <n>` | 生成 token 数,默认 8 |
| `--ctx <n>` | 覆盖上下文长度(GGUF 默认截到 2048) |
| `--threads <n>` | CPU 线程数,0 = 硬件线程 |
| `--backend cpu\|cuda` | 执行后端,默认 cpu |
| `--device <n>` | CUDA 设备编号 |
| `--temperature` / `--top-k` / `--seed` | 采样参数 |
| `--template qwen2\|plain` | 聊天模板 |
| `--verbose` | bench 时打印每步 top-5 logits |

## 阶段完成情况

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0–2 | 构建、Tensor、基础算子 | ✅ 已验证 |
| 3–5 | Tiny loader、tokenizer、KV cache | ✅ 已验证 |
| 6–8 | Transformer 前向、greedy 生成、文档 | ✅ 已验证 |
| 9–10 | prefill/decode 统一、采样、多轮聊天 | ✅ 已验证 |
| 11–13 | GGUF reader、模型映射、BPE tokenizer | ✅ 已验证(真实 Qwen2-0.5B) |
| 14–15 | Q8_0 / Q4_0 / Q4_1 量化 | ✅ 已验证 |
| 16 | CPU 线程池 | ✅ 已验证(数值一致,真实模型约 3× 加速) |
| 17 | Benchmark | ✅ 已验证 |
| 18–20 | CUDA 后端 | ⚠️ 代码完成,未在 GPU 上编译运行 |
| 21 | 文档、功能对照、CLI 冒烟 | ✅ 已完成 |

## 测试

`ctest` 运行两个测试:13 组 C++ 单元测试与 1 个 Python CLI 冒烟测试。

| 测试 | 覆盖内容 |
|---|---|
| `test_tensor` | 构造、索引、reshape、越界 |
| `test_ops` | MatMul/Linear/RMSNorm/Softmax/SwiGLU/RoPE |
| `test_thread_pool` | 分块唯一性、异常回传、重复调用、1 vs 4 线程一致 |
| `test_kv_cache` | 读写、层间隔离、越界 |
| `test_forward` | 单步前向、批量等价、错误路径 |
| `test_loader` | manifest 校验、截断文件 |
| `test_tokenizer` | BPE 合并、特殊 token、字节回退、解码 |
| `test_sampler_chat` | 贪心/温度/top-k、模板拼接 |
| `test_benchmark` | 计数、计时、空 prompt |
| `test_quant` | block 布局、不对称 block、尾部 block、误差 |
| `test_gguf` | v2/v3 解析、metadata、错误文件 |
| `test_gguf_loader` | 形状映射、GQA、量化类型、前向 |
| `test_cuda_runtime` | CPU-only 错误路径 / 设备信息 |
| `cli-smoke` | generate、bench、参数校验、cuda 拒绝、inspect |

## 性能参考

Qwen2-0.5B-Instruct-Q8_0、`--ctx 256`、GTX 1650 机器上的 CPU 路径:

| 线程 | decode 吞吐 | 说明 |
|---|---|---|
| 1 | 0.10 token/s | 与 4 线程的 top-5 logits 完全一致 |
| 4 | 0.30 token/s | 约 3× 加速,数值结果不变 |

Tiny 模型矩阵很小,多线程可能持平或略慢,属于正常现象;量化 GEMV 是主要瓶颈。

## CUDA 构建

```powershell
cmake -S . -B build-cuda -DMINI_LLA_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-cuda --config Release

.\build-cuda\Release\mini-lla.exe --cuda-info
.\build-cuda\Release\mini-lla.exe bench model.gguf -p "Hello" -n 16 --ctx 256 --backend cuda
```

`75` 对应 GTX 1650;RTX 30 系列用 `86`,40 系列用 `89`。需要 CUDA Toolkit 12.x。

**验证状态说明**:开发机只有驱动(592.82)与 GTX 1650(算力 7.5),没有安装
CUDA Toolkit,因此 `.cu` 文件从未经过编译器,GPU 路径也从未运行。CPU 构建会编译
全部 `.cc` 桩,`--backend cuda` 会给出明确错误。启用 CUDA 后建议按以下顺序验证:

1. `--cuda-info` 能否枚举设备;
2. `inspect --backend cuda` 检查权重上传数量与显存占用;
3. `bench --verbose --backend cuda` 与 CPU 的 top-5 logits 对照;
4. 固定 prompt 连续生成,确认 KV cache 复位与边界正确。

## 已知限制

- GGUF 上下文长度在加载时截到 2048,可用 `--ctx` 调整(注意 KV cache 内存)。
- 分词只做"空格 + 单词"的预切分,未实现 Qwen2 的完整正则预切分。
- 不支持 F16、K-quants、IQ-quants、多 GPU、FlashAttention 与任意 Jinja2 模板。
- 多轮聊天每轮重新 prefill 完整历史,尚未复用 KV cache。
- cuBLAS 句柄为进程级静态对象,当前只支持单设备。

## 目录结构

```text
include/mini_llama/   27 个头文件:接口与数据布局
src/                  31 个实现文件,含 3 个 CUDA 内核
tests/                13 组测试 + GGUF fixture
scripts/              Tiny 模型导出、CLI 冒烟测试
models/tiny/          Tiny manifest 与随机权重
```

建议阅读顺序:`tensor.h` → `ops.cc` → `loader.cc` → `forward.cc` → `gguf.cc` →
`gguf_tokenizer.cc` → `quant.cc` → `thread_pool.cc` → `cuda_forward.cc`。

## 代码规范

- Google C++ 风格:文件 `snake_case`、函数与类 `PascalCase`、成员带尾下划线、
  常量 `kPascalCase`、2 空格缩进、每行不超过 80 列。
- 每个类和函数都有中文注释;复杂流程拆成小函数。
- 源码为 UTF-8,MSVC 必须加 `/utf-8`(CMake 已配置)。

## 许可

仓库目前尚未添加 `LICENSE` 文件;如需开源发布,请先补充许可证(例如 MIT)。
