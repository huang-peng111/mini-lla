# mini-lla

这个仓库是拿来"搞明白大模型怎么吐字"的。

很多人第一次接触推理,看到的都是 PyTorch 或者 llama.cpp 那种几万行的工程,函数套函数,
宏套宏,读完还是不知道一个 token 到底怎么变成下一个 token。这个项目从最基础的地方
重新搭了一遍:一个 F32 张量怎么写、矩阵乘怎么算、KV cache 怎么存、GGUF 怎么解析、
量化 block 怎么摆、线程池怎么切任务,全部摊开放在你能读到的地方。

所以你在这里看不到框架封装,也没有 `import torch`。整个推理链路是我自己用 C++17
写的,只依赖标准库和 CMake。CUDA 部分额外依赖 NVIDIA 的运行时和 cuBLAS。

再说得直白点:这是个学习项目,不是生产级推理引擎。它可以加载真实的 Qwen2 模型并
生成连贯的英文,但速度、格式覆盖和稳定性都比不上 llama.cpp。它的价值在于"每一行
你都能读懂,而且能动手改"。

## 它能做什么,不能做什么

目前能做的事:

- 加载两种模型:仓库自带的 Tiny 模型(JSON manifest 加裸 F32 权重),以及真实的
  GGUF v2/v3 文件。
- 跑通完整的 Transformer 前向,支持 GQA、RoPE 的两种配对方式(LLaMA 和 NeoX)、
  Qwen2 的 QKV 偏置,以及 tied lm_head。
- 用 GGUF 里自带的词表做字节级 BPE 分词,支持 merges、特殊 token 和字节回退。
- 加载 Q8_0、Q4_0、Q4_1 三种量化权重,block 布局和 ggml 一致。
- 用多线程跑 CPU 推理,并提供 benchmark 看 prefill 和 decode 的速度。
- 跑多轮聊天,内置 Qwen2 的 `<|im_start|>` 模板。
- 可选 CUDA 后端:权重常驻显存,支持 cuBLAS F32 线性层和量化 GEMV。

目前做不了的事:

- 不支持 F16、K-quants、IQ-quants 这些格式,加载时会直接报错。
- 不支持多 GPU、FlashAttention,也没有在服务端跑的 API。
- 不做训练,也没有反向传播。
- 不是所有模型都能跑,目前只覆盖 LLaMA 和 Qwen2 这一套 tensor 命名。

## 先把项目跑起来

### 你需要装什么

Windows 上最省事的组合是 Visual Studio 2022(勾选"使用 C++ 的桌面开发")加 CMake。
Python 3 是可选的,只有生成 Tiny 模型和跑 CLI 冒烟测试时才用得到。

CUDA 默认是关的,不装也能编译。只有你想试 GPU 那条路,才需要 NVIDIA 显卡和
CUDA Toolkit 12.x。

### 三条命令跑通

在项目根目录打开 PowerShell:

```powershell
cmake -S . -B build
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

第一条生成构建文件,第二条编译,第三条跑测试。测试必须在这种项目根目录下执行,
因为用例会用相对路径去找 `models/tiny`。

如果三条都过了,接着跑一次生成:

```powershell
.\build\Debug\mini-lla.exe generate --model models\tiny -p "hello" -n 8
```

### 你应该看到什么

正常输出大概是这样:

```text
model: models\tiny
backend: cpu
threads: 8
weight bytes: 115328
prompt tokens: 6
output: hello...
```

`threads: 8` 是我这台机器的硬件线程数,你那边可能是 4 或者 16,不影响结果。
`prompt tokens: 6` 是因为 ASCII tokenizer 会在开头补一个 BOS,`hello` 五个字母
再加一个 BOS 正好是 6。

至于 `output` 后面那串看起来像乱码的东西,别怀疑人生。Tiny 模型用的是随机权重,
它只负责验证"数据能从输入流到输出",不负责说人话。想看正常回答,得用真实模型。

## 常用命令,以及它们实际在干什么

### generate

单次生成。把 prompt 编码成 token,先一次性算出所有 prompt token 的 KV cache
(这一步叫 prefill),然后每次采样一个 token 再喂回去(这一步叫 decode)。

```powershell
.\build\Debug\mini-lla.exe generate --model models\tiny -p "hello" -n 8
.\build\Debug\mini-lla.exe generate --model models\tiny -n 8 --temperature 0.8 --top-k 8 --seed 42
```

`--temperature 0` 是贪心,每次取概率最大的 token;给它一个大于 0 的值才会随机。
`--top-k` 只在温度大于 0 时起作用。`--seed` 固定之后,同样的输入每次都会得到
同样的输出,做对比实验时很有用。

### run

多轮聊天,一直读输入直到你敲 `/exit`。支持 `/help` 和 `/clear`。

```powershell
.\build\Debug\mini-lla.exe run models\tiny --temperature 0.8 --top-k 8 --seed 42
```

这里的实现比较简单:每一轮都把完整历史重新 prefill 一遍。好处是逻辑清楚、不容易
出 bug,代价是聊得越久越慢。做前缀缓存复用是后面很值得动手的一件事。

### bench

只测速度,不关心回答好不好。它分别统计 prefill 和 decode 的耗时,再算 token/s。

```powershell
.\build\Debug\mini-lla.exe bench models\tiny -n 4 --threads 4
.\build\Debug\mini-lla.exe bench model.gguf -p "Hello" -n 16 --threads 4 --verbose
```

加 `--verbose` 会把每一步的 top-5 logits 打出来,调数值问题的时候很有用。

### inspect 和 inspect-gguf

`inspect` 看模型本身:层数、维度、head 数、rope 参数、权重占用,以及每种量化格式
各有多少个线性层。

`inspect-gguf` 只看文件结构:版本、metadata 和 tensor 表。它不加载权重,所以很快,
适合先确认"这个文件里到底有什么"。

```powershell
.\build\Debug\mini-lla.exe inspect models\tiny
.\build\Debug\mini-lla.exe inspect-gguf model.gguf
```

### --cuda-info

看看当前构建和机器上的 CUDA 状态。

```powershell
.\build\Debug\mini-lla.exe --cuda-info
```

CPU 构建会直接告诉你 `CUDA: 未启用`,不会假装自己能跑。

## 参数速查

| 参数 | 作用 | 备注 |
|---|---|---|
| `--model <path>` | 模型路径 | 可以是 Tiny 目录,也可以是 `.gguf` 文件 |
| `-p, --prompt <text>` | 输入文本 | 默认 `hello` |
| `-n, --n-predict <n>` | 生成多少个 token | 默认 8,`--n` 也能用 |
| `--ctx <n>` | 覆盖上下文长度 | GGUF 默认截到 2048 |
| `--threads <n>` | CPU 线程数 | 0 表示用满硬件线程 |
| `--backend cpu\|cuda` | 选后端 | 默认 cpu |
| `--device <n>` | CUDA 设备编号 | 不写就是 0 |
| `--temperature <t>` | 采样温度 | 0 表示贪心 |
| `--top-k <k>` | 候选数量 | 0 表示不限制 |
| `--seed <s>` | 随机种子 | 0 表示每次随机 |
| `--template qwen2\|plain` | 聊天模板 | 不写就按模型自动选 |
| `--verbose` | 打印每步 top-5 | 只在 bench 里有意义 |

模型路径也可以直接写在命令后面,不用带 `--model`,两种写法效果一样。

## 拿真实模型跑一次

仓库里只有 Tiny 模型。想让它说人话,得自己准备一个 GGUF 文件。我测试用的是
`Qwen2-0.5B-Instruct-Q8_0.gguf`,文件大概 531 MB,加载后权重占用约 500.9 MB。

```powershell
# 先把路径换成你自己的文件
.\build\Debug\mini-lla.exe generate model.gguf -p "Hello, my name is" -n 8 --ctx 256 --threads 4
```

我这边实际跑出来的结果:

```text
prompt tokens: 5
output: Hello, my name is Sarah and I am a
```

`prompt tokens: 5` 说明 "Hello, my name is" 被切成了 5 个 token,和 llama.cpp
的切法一致。

这个模型的 inspect 结果大概是这样:

```text
vocab_size:   151936
dim:          896
hidden_dim:   4864
n_layers:     24
n_heads:      14
n_kv_heads:   2
head_dim:     64
max_seq_len:  2048
rope_theta:   1e+06
rope_type:    neox
tied_lm_head: true
weight bytes: 525120000
linear weights: F32=0 Q8_0=169 Q4_0=0 Q4_1=0
```

`n_heads` 是 14,`n_kv_heads` 是 2,这就是大家常说的 GQA:每 7 个 query head
共用一组 key/value。`tied_lm_head: true` 说明这个模型没有单独的 `output.weight`,
输出投影直接复用词表嵌入。

### 关于内存

加载模型时上下文长度会被截到 2048,这是故意的。Qwen2-0.5B 的元数据里写着 32768,
如果按原值分配 KV cache,光缓存就要几百 MB。截到 2048 之后,24 层、2 个 kv head、
head_dim 64 的 KV cache 大约 50 MB(算上 key 和 value 两份)。

如果机器内存紧张,把 `--ctx` 调小,比如 256 或 512。反过来,如果你确信内存够,
也可以往大了调。

## 代码该从哪读起

别一上来就扎进 GGUF 解析,那里全是二进制细节,容易劝退。按下面这个顺序走,每一步
都能跑:

1. `tensor.h` 和 `tensor.cc`:先搞清楚 shape 和行主序,后面所有东西都建立在这上面。
2. `ops.cc`:矩阵乘、RMSNorm、Softmax、SiLU、RoPE,都是几十行的短函数。
3. `loader.cc`:看 Tiny 模型怎么从 manifest 里读出来。
4. `forward.cc`:这是整个项目的核心,一个 token 进,logits 出。建议读三遍,第一遍
   看结构,第二遍追每个张量的形状,第三遍看 GQA 怎么映射。
5. `gguf.cc` 和 `gguf_loader.cc`:真实模型的二进制格式和映射规则。
6. `gguf_tokenizer.cc`:字节级 BPE,顺便能理解为什么一个 token 可能只有半个汉字。
7. `quant.cc`:三种量化格式的 block 布局和反量化。
8. `thread_pool.cc`:常驻线程池,重点看 `epoch_` 和原子领取分块。
9. `cuda_forward.cc`:如果前八步都看懂了,这里就是把同一套流程搬到 GPU。

整个仓库最大的源文件也就 355 行(`main.cc`),大多数文件只有一两百行,最短的
`chat.cc` 不到 10 行。函数也拆得比较碎,超过五十行的很少,读起来不会太累。

## 项目完成到哪一步

项目按 22 个阶段推进(0 到 21):

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0-2 | 构建、Tensor、基础算子 | 已验证 |
| 3-5 | Tiny loader、tokenizer、KV cache | 已验证 |
| 6-8 | 前向、greedy 生成、文档 | 已验证 |
| 9-10 | prefill/decode 统一、采样、多轮聊天 | 已验证 |
| 11-13 | GGUF reader、模型映射、BPE | 已验证 |
| 14-15 | Q8_0、Q4_0、Q4_1 | 已验证 |
| 16 | CPU 线程池 | 已验证 |
| 17 | benchmark | 已验证 |
| 18-20 | CUDA 后端 | 已完成 |
| 21 | 文档与收尾 | 已完成 |

## 测试

`ctest` 会跑两个东西:一个是 C++ 单元测试,另一个是 Python 写的 CLI 冒烟测试
(找不到 Python 时会跳过)。

C++ 测试一共 13 组,按功能拆文件:

| 文件 | 测什么 |
|---|---|
| `test_tensor` | 构造、索引、reshape、越界 |
| `test_ops` | MatMul、Linear、RMSNorm、Softmax、SwiGLU、RoPE |
| `test_thread_pool` | 分块唯一性、异常回传、重复调用、1 线程和 4 线程结果一致 |
| `test_kv_cache` | 读写、层间隔离、越界 |
| `test_forward` | 单步前向、批量等价、错误路径 |
| `test_loader` | manifest 校验、截断文件 |
| `test_tokenizer` | BPE 合并、特殊 token、字节回退、解码 |
| `test_sampler_chat` | 贪心、温度、top-k、模板拼接 |
| `test_benchmark` | 计数、计时、空 prompt |
| `test_quant` | block 布局、不对称 block、尾部 block、误差 |
| `test_gguf` | v2/v3 解析、metadata、错误文件 |
| `test_gguf_loader` | 形状映射、GQA、量化类型、前向 |
| `test_cuda_runtime` | CPU 构建下的错误路径,或者设备信息 |

想单独跑某个测试也可以:

```powershell
.\build\Debug\mini-lla-tests.exe
```

它会把 13 组依次跑一遍,哪组挂了会打印文件和行号。

## 性能数字是怎么来的

我在 GTX 1650 那台机器上,用 Qwen2-0.5B-Instruct-Q8_0、`--ctx 256`、Debug 构建
测了一组:

| 线程数 | decode 速度 | 说明 |
|---|---|---|
| 1 | 0.10 token/s | 与 4 线程的 top-5 logits 完全一致 |
| 4 | 0.30 token/s | 大约 3 倍,数值结果不变 |

每秒 0.3 个 token 确实很慢。这是单线程工作、没有 SIMD 优化的结果,量化 GEMV
基本就是内存带宽在拖后腿。想看明显提升,下一步是给量化线性层加 AVX2,再往上才是
CUDA。

顺便说一句,Tiny 模型上多线程反而可能更慢。矩阵太小了,线程同步的开销比计算本身
还大,这属于正常现象,不是线程池写坏了。

## CUDA 后端

仓库里有一套完整的 CUDA 路径:设备枚举、显存 RAII、权重上传、cuBLAS 的 F32
线性层、Q8_0/Q4_0/Q4_1 的 GEMV 内核、GPU 上的 RMSNorm、RoPE、SiLU、softmax,
还有 GPU KV cache 和 attention。前向过程基本都留在显卡上,只有最后的 logits
会拷回 CPU 做采样。

默认构建不启用 CUDA,这样没装 Toolkit 的机器也能直接编译。CPU 构建下,这些接口
是直接抛错的桩,不会静默出错:

CPU 构建下,这些接口会变成直接抛错的桩,所以不会静默出错:

```text
error: CUDA 未启用: 请使用 -DMINI_LLA_CUDA=ON 重新构建
```

启用 GPU 路径需要 CUDA Toolkit 12.x:

```powershell
cmake -S . -B build-cuda -DMINI_LLA_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-cuda --config Release
```

`75` 对应 GTX 1650 的算力。RTX 30 系列用 `86`,40 系列用 `89`,不确定的话可以先
跑 `nvidia-smi --query-gpu=compute_cap --format=csv` 看一眼。

构建完成后可以这样检查:

1. `--cuda-info` 看看能不能枚举出设备;
2. `inspect --backend cuda` 看权重上传了多少、占多少显存;
3. `bench --verbose --backend cuda` 和 CPU 的 top-5 logits 对一下;
4. 用固定 prompt 连续生成几句,确认 KV cache 没有串位置。

如果哪一步对不上,大概率是 kernel 里的下标或者 quantized 权重的布局问题。这部分
就留给你当练习了。

## 踩过的坑

这些是我实际遇到过的问题,写下来省得你再花时间。

**Tiny 模型输出乱码。** 随机权重就是这样,别修。它的输出没有语义,判断正确性要看
logits 形状、数值是否有限、同样输入是否得到同样结果。

**中文注释导致编译报错。** MSVC 默认不按 UTF-8 解析源码,中文注释的最后一个字节
可能被当成转义字符,然后报一堆莫名其妙的语法错误。CMake 里已经加了 `/utf-8`,
如果你自己写构建脚本,记得带上。

**传了负数的 `-n`。** 早期版本会悄悄不生成任何 token,现在会直接报错:
`--n-predict must be in [0, 4096]`。

**上下文不够用。** GGUF 默认截到 2048,长对话会提示历史超长。要么 `/clear`,
要么用 `--ctx` 调大,但注意 KV cache 会跟着涨。

**没有 CUDA 却写了 `--backend cuda`。** 会直接告诉你 CUDA 未启用,不会崩。

## 目录结构

```text
include/mini_llama/   27 个头文件,接口和数据结构都在这里
src/                  31 个实现文件,包括 3 个 CUDA 内核
tests/                13 组测试和一个生成 GGUF 的测试夹具
scripts/              Tiny 模型导出脚本、CLI 冒烟测试
models/tiny/          Tiny 模型(manifest 加随机权重)
```

头文件和源文件基本是一一对应的,看头文件知道"有什么",看源文件知道"怎么做的"。

## 代码风格

整体按 Google C++ 风格写:文件名用 `snake_case`,函数和类用 `PascalCase`,
成员变量带尾下划线,常量用 `k` 开头,两空格缩进,每行尽量不超过 80 列。

每个类和函数都有中文注释,复杂的流程会拆成小函数。写这些注释的目的不是应付检查,
而是过几个月再回来看时,能快速想起当时为什么这么写。

## 许可

仓库目前还没有 `LICENSE` 文件。如果打算开源发布,建议先补一份,比如 MIT 或者
Apache-2.0,这个需要仓库所有者来决定。
