# mini-lla:从零复刻的 LLaMA 风格推理程序

这是按 `IMPLEMENTATION_ROADMAP.md` 逐阶段重写的 C++17 推理程序。当前已完成
**阶段 0 到阶段 15**:CPU Tiny 模型闭环、GGUF reader/loader/tokenizer、量化
(Q8_0 / Q4_0 / Q4_1)以及对应的单元测试。

## 代码规范

- 采用 Google C++ 风格:文件名 `snake_case`、函数与类 `PascalCase`、成员变量
  带尾下划线、常量 `kPascalCase`、命名空间 `mini_lla`、缩进 2 空格、每行不超过
  80 列。
- 每个类和函数都有中文注释,说明它做什么、参数与返回值是什么。
- 单个函数尽量控制在 40 行以内,复杂流程拆成小函数。
- 源码是 UTF-8 且包含中文注释,MSVC 编译必须带 `/utf-8`(CMake 已配置)。

## 构建与测试

```powershell
cmake -S . -B build
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

`ctest` 会运行两个测试:自带 runner 的 C++ 单元测试,以及一份 Python CLI 冒烟
测试(需要 Python 3,找不到时会自动跳过)。

## 命令

```powershell
# 单次生成(tiny 模型是随机权重,只用于验证链路)
.\build\Debug\mini-lla.exe generate --model models\tiny -p "hello" -n 8

# 固定种子的温度 / top-k 采样
.\build\Debug\mini-lla.exe generate --model models\tiny -n 8 --temperature 0.8 --top-k 8 --seed 42

# 多轮聊天,命令:/help /clear /exit
.\build\Debug\mini-lla.exe run models\tiny --temperature 0.8 --top-k 8 --seed 42

# 查看模型配置和权重统计
.\build\Debug\mini-lla.exe inspect models\tiny

# 查看 GGUF 的 metadata 与 tensor 表
.\build\Debug\mini-lla.exe inspect-gguf path\to\model.gguf

# 真实模型:GGUF 路径可以直接传给 generate / run / inspect
.\build\Debug\mini-lla.exe generate path\to\Qwen2-0.5B-Instruct-Q8_0.gguf -p "Hello, my name is" -n 6 --ctx 256
.\build\Debug\mini-lla.exe run path\to\Qwen2-0.5B-Instruct-Q8_0.gguf -n 32 --ctx 512
```

常用选项:`--model`、`-p/--prompt`、`-n/--n/--n-predict`、`--ctx`、
`--temperature`、`--top-k`、`--seed`、`--template qwen2|plain`。
模型路径也可以写成位置参数,例如 `inspect models/tiny`。

## 支持的模型格式

| 项目 | 支持情况 |
|---|---|
| Tiny 格式 | `model.json` + `model.bin`(manifest 记录每个 tensor 的偏移和字节数) |
| GGUF 版本 | v2 / v3 |
| GGUF tensor 类型 | F32、Q8_0、Q4_0、Q4_1(与 ggml 的 block 布局一致) |
| 架构 | LLaMA / Qwen2 命名(`blk.N.*`),Qwen2 使用 NeoX 风格 RoPE |
| 分组查询注意力 | 支持,`n_kv_heads` 小于 `n_heads` 时按组共享 KV |
| 注意力偏置 | 支持 `attn_q/k/v.bias` |
| 权重共享 | `output.weight` 缺失时自动与 `token_embd.weight` 共享 |
| tokenizer | GGUF 内置词表 + merges,字节级 BPE,支持特殊 token 与字节回退 |
| 聊天模板 | 内置 Qwen2 (`<|im_start|>`)和通用文本模板;任意 Jinja2 模板不支持 |

未支持:F16、K-quants、IQ-quants、多 GPU、SIMD 与多线程(属于阶段 16 之后)。

## 已实现的阶段与验证方式

- 阶段 1–2:张量与算子 —— `test_tensor.cc`、`test_ops.cc`。
- 阶段 3:tiny manifest 与权重加载 —— `test_loader.cc`,校验形状与字节数。
- 阶段 4:tokenizer —— `test_tokenizer.cc`。
- 阶段 5:KV cache —— `test_kv_cache.cc`。
- 阶段 6–7:单 token 前向与贪心生成 —— `test_forward.cc`、CLI 冒烟测试。
- 阶段 9:prefill / decode 统一输入 —— `MiniBatch`,批量与逐 token 结果一致。
- 阶段 10:多轮聊天 —— `test_sampler_chat.cc` 与 `run` 命令。
- 阶段 11:GGUF reader —— `test_gguf.cc`(含 magic / 版本 / 截断错误路径)。
- 阶段 12:GGUF 映射到模型 —— `test_gguf_loader.cc`,合成 GGUF 跑通前向。
- 阶段 13:GGUF tokenizer —— `test_tokenizer.cc`;真实模型 `generate -n 1` 冒烟。
- 阶段 14–15:Q8_0 / Q4_0 / Q4_1 —— `test_quant.cc`,覆盖不对称 block、尾部
  block、奇偶 nibble 与误差上界。

## 已知限制

- 加载 GGUF 时上下文长度按 2048 截断(元数据里常见 32768),用 `--ctx` 调整,
  这样可以限制 KV cache 的内存占用。
- 分词只做了"空格 + 单词"的预切分,没有实现 Qwen2 的完整正则预切分,长句
  标点边界与 llama.cpp 可能有细微差异。
- 量化线性层是逐 block 即时反量化,没有 SIMD 优化,所以真实模型上每个 token
  需要数秒;线程池与 SIMD 属于阶段 16。
- Tiny 模型使用随机权重,只用来验证推理链路,生成内容没有语义。

## 目录结构

```text
include/mini_llama/   接口:张量、算子、量化、模型、GGUF、tokenizer、聊天
src/                  实现,与头文件一一对应
tests/                自带 runner 的单元测试 + GGUF fixture 生成器
scripts/              tiny 模型导出、CLI 冒烟测试
models/tiny/          Tiny manifest 与 F32 权重
```
