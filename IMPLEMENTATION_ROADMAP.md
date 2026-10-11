# mini-llama.cpp 从零复刻实施路线

> 目标：在**独立的新文件夹**里，从空项目开始实现一个结构和运行方式与参考项目相同、但每阶段只包含必要功能的 C++17 LLaMA 风格推理程序。完成路线后再逐步补齐 CPU、GGUF、量化、聊天、性能优化和可选 CUDA 能力。
>
> 本文档只定义实施流程。本阶段不在参考仓库内改代码，也不在新目录中提前创建程序源码。

## 1. 完成目标与约定

### 最终目标

- 使用 C++17、CMake；核心实现使用普通 C++，CUDA 部分使用 `.cu`。
- 目录层级与参考项目一致：`include/mini_llama` 放接口，`src` 放实现，`models` 放模型数据，`scripts` 放必要辅助脚本，`tests` 放测试。
- 先完成 CPU Tiny 模型的完整推理闭环，再逐项加入更真实的模型格式和优化。
- 可用命令行加载模型、生成文本、进行多轮终端聊天、查看模型信息并运行基础 benchmark。
- 量化模型可在 CPU 上执行；选定算子可选走 NVIDIA CUDA 路径。默认 CPU 构建不需要 CUDA Toolkit。
- 每一阶段都有可运行产物和清晰验收条件；没有必要的脚本、测试框架或抽象暂不添加。

### 新手实施约定

- 推荐在 Windows 上使用 Visual Studio 2022 C++ 工作负载、CMake、Git 和 Python 3；优先使用 PowerShell 命令。Linux/macOS 后续按需验证。
- 初期模型使用小尺寸、确定性生成的随机权重，只用来确认程序链路正确；随机权重**不会产生有意义的聊天答案**。
- 第一阶段先支持 JSON manifest + F32 权重 + 字符级词表。真实对话效果等到 GGUF 阶段再验证。
- Python 只用于生成微型模型文件和少数 CLI 回归检查；基础推理和单元测试保持 C++。
- 不先实现多线程、CUDA、量化、批处理、前缀缓存或复杂调试系统。每项能力进入路线时再添加对应代码和测试。
- 复刻以重新编写、理解和验证为目的；参考项目用于核对接口语义、张量名和行为，不把整份实现拷贝进新项目。

## 2. 新项目目录约定

项目根目录单独创建为 `mini-llama-cpp-replica/`。逐阶段长成下列结构；尚未进入路线的目录可先留空或不创建内部文件。

```text
mini-llama-cpp-replica/
  CMakeLists.txt
  README.md
  LICENSE
  .gitignore
  include/
    mini_llama/
      tensor.h
      ops.h
      model.h
      tokenizer.h
      kv_cache.h
      context.h
      forward.h
      sampler.h
      loader.h
      chat.h
      prompt_builder.h
      terminal.h
      debug.h
      gguf.h
      gguf_loader.h
      gguf_tokenizer.h
      quantized_tensor.h
      quant.h
      thread_pool.h
      matmul_dispatch.h
      backend.h
      cuda_runtime.h
      cuda_tensor.h
      cuda_matmul.h
      cuda_quant.h
      cuda_ops.h
      cuda_kv_cache.h
      cuda_attention.h
  src/
    tensor.cc
    ops.cc
    model.cc
    tokenizer.cc
    json_vocab_tokenizer.cc
    kv_cache.cc
    context.cc
    forward.cc
    sampler.cc
    loader.cc
    main.cc
    chat.cc
    prompt_builder.cc
    terminal.cc
    debug.cc
    gguf.cc
    gguf_loader.cc
    gguf_tokenizer.cc
    quant.cc
    quantized_tensor.cc
    thread_pool.cc
    matmul_optimized.cc
    backend.cc
    cuda_runtime.cc
    cuda_tensor.cc
    cuda_matmul.cc
    cuda_quant.cc
    cuda_quant.cu
    cuda_ops.cc
    cuda_ops.cu
    cuda_kv_cache.cc
    cuda_kv_cache.cu
    cuda_attention.cc
    cuda_attention.cu
  models/
    tiny/
      model.json
      model.bin
      vocab.json
  scripts/
    export_tiny_model.py
    test_cli.py
  tests/
    test_main.cc
    test_tensor.cc
    test_ops.cc
    test_loader.cc
    test_tokenizer.cc
    test_kv_cache.cc
    test_forward.cc
    test_sampler.cc
```

目录树是最终形态的参照，不是第一天就要创建一堆空文件。GGUF、量化、线程池和 CUDA 文件分别在对应阶段新建。最终 `scripts` 和 `tests` 只保留有实际用途的文件；不复制参考仓库里重复的 GoogleTest 镜像、测试 fixtures 或覆盖同一路径的许多 CLI 脚本。

## 3. 分阶段实施路线

每个阶段按顺序进行。某阶段的验收未通过前，不开始依赖它的下一阶段。阶段中的接口以能支持当前目标为准，避免提前设计复杂扩展点。

### 阶段 0：建立独立项目与最小构建

**小目标：** 新文件夹能配置、编译并运行一个打印版本信息的 C++ 程序。

- 创建新的 Git 仓库或独立项目目录；不复制参考项目的 `src`、`include`、测试和模型文件。
- 写最小 `CMakeLists.txt`，指定 C++17，并建立 `mini-llama` 可执行文件目标。
- 增加 `.gitignore` 忽略 `build/`、IDE 缓存和 Python 缓存；README 写 Windows 构建步骤。
- 初始只提交构建入口、入口程序、README 和许可证。

**验收：** 在 PowerShell 执行 `cmake -S . -B build`、`cmake --build build`，运行 `build/Debug/mini-llama.exe` 能看到版本/欢迎文字。Release 配置也能构建。

### 阶段 1：F32 Tensor 与最小测试框架

**小目标：** 有一个可检查形状的连续内存 F32 Tensor，测试能报告通过或失败。

- `Tensor` 保存 `shape` 和 `std::vector<float> data`；实现元素数、构造、reshape、二维访问与 shape 校验。
- 校验维度为正、元素数乘法不溢出、访问不越界、reshape 前后元素总数一致。
- 用极小的自定义 C++ 测试 runner 注册并运行测试，不引入外部测试框架。
- 添加 `test_tensor.cc`，只覆盖构造、索引、reshape 和错误输入。

**验收：** Debug/Release 都能通过 Tensor 测试；空 shape、错误索引和不匹配 reshape 都以明确异常失败，不发生越界访问。

### 阶段 2：CPU 基础算子

**小目标：** 独立验证前向计算需要的数学基础。

- 实现朴素 `MatMul`/`Linear`、`RmsNorm`、`Softmax`、`Rope`、`Silu`、`SwiGlu`、逐元素加法/乘法和 `ArgMax`。
- 每个算子先只支持前向路径、F32 和模型实际使用的 shape；不实现反向传播或通用计算图。
- 为矩阵维度、输入 shape、softmax 空输入和非法参数添加必要检查。
- 添加 `test_ops.cc`，用手算小矩阵和特殊值验证结果。

**验收：** 矩阵乘、归一化、softmax、RoPE、激活函数和 argmax 的小例子均与手算结果在约定误差内一致；非法 shape 有清楚报错。

### 阶段 3：Tiny 配置、权重布局和 loader

**小目标：** 从 `models/tiny` 读取模型配置和 F32 权重，并检查每个 tensor。

- 定义 `ModelConfig`、`LayerWeights` 和 `MiniLlamaModel`；第一版只保存 embedding、Transformer 层权重、norm 和 lm_head。
- manifest 使用 `model.json`，列出模型维度、权重名称、shape、dtype、offset 和 byte size；权重按小端 F32 顺序存在 `model.bin`。
- 实现 Tiny loader：检查必需字段、张量是否齐全、shape/byte size 是否匹配，以及权重文件截断/缺失情况。
- 添加标准库 Python `export_tiny_model.py`，用固定 seed 生成小模型、JSON manifest 和 F32 二进制权重；不依赖 NumPy或模型下载。
- 添加 `test_loader.cc` 验证正常加载和最常见的损坏文件。

**验收：** 生成两次的模型字节完全一致；C++ loader 能加载模型并检查关键 tensor shape；缺失文件、错误维度、短文件和不支持 dtype 均给出可读错误。

### 阶段 4：Tokenizer 与文本到 token 的映射

**小目标：** prompt 可以转成 token id，生成 token 可以还原为文本。

- 定义 `ITokenizer` 的最小接口：Encode、DecodeToken、Decode、词表大小、BOS/EOS/UNK id。
- 先支持 Tiny `vocab.json` 的最长匹配/字符级映射；如需更简化，先用 ASCII 字符 tokenizer 跑通，再切换 JSON 词表。
- 规定 BOS 由 tokenizer 添加、EOS 用于停止；约定控制 token 的文本显示行为。
- 添加 `test_tokenizer.cc` 覆盖普通字符、空字符串、未知字符、特殊 token 和往返转换。

**验收：** prompt 可以稳定编码，生成 token 可逐个解码；超出词表的输入按明确策略落到 UNK，不访问无效 id。

### 阶段 5：KV Cache 与推理 Context

**小目标：** 为每层每个位置保存 Attention 的 key/value，并拒绝越界写入。

- 定义 CPU `KvCache`，存储形状为 `[layers, sequence, kv_heads, head_dim]` 的 keys 和 values。
- 实现按 layer/position 写入与按 head 读取；读写 shape、position 和 layer 都检查边界。
- 定义 `MiniLlamaContext`，持有本次请求的 KV cache、当前位置和已处理 token 数；Context 与模型配置匹配。
- 添加 `test_kv_cache.cc` 覆盖写读、层间隔离和位置越界。

**验收：** 不同层/位置/head 的缓存值互不覆盖；最大上下文之外的写入能报告清楚错误。

### 阶段 6：单 token Transformer 前向

**小目标：** 输入一个 token，计算出词表大小的 logits。

- 按 LLaMA 风格顺序实现 embedding lookup、Attention RMSNorm、Q/K/V projection、RoPE、causal attention、KV cache 写入、输出 projection/residual、FFN RMSNorm、SwiGLU FFN、最终 norm 和 lm_head。
- 第一版用单 token `ForwardToken`；KV cache 中只能读取当前位置及之前的数据，形成 causal mask 效果。
- 每个子计算用清楚的函数分段；错误信息包含所在层和预期/实际 shape。
- 添加 `test_forward.cc`，用小型固定权重模型确认 logits shape、有限值和 KV cache position。

**验收：** 同一个固定模型和 token 总是产生相同 logits；第 0 个位置能推理，继续输入 token 后 cache 位置递增；超出上下文上限立即失败。

### 阶段 7：Greedy 生成命令

**小目标：** 完成 `prompt → tokens → logits → next token → 文本` 的端到端闭环。

- 实现 `mini-llama generate`；第一版只提供模型目录、prompt、预测 token 数和 help 参数。
- 先对 prompt 所有 token 做 prefill；每轮取最大 logit 作为下一个 token，再把它送回 forward 继续 decode。
- 达到预测数量或采到 EOS 时停止；生成 token 逐个解码输出。
- 检查负数 token 数、空/超长 prompt、模型缺失和无效 token id；标准错误打印原因并返回非零状态码。
- 增加一个短的 `scripts/test_cli.py`，从已编译程序运行固定 prompt，检查返回码和输出结构；不比对随机模型的自然语言质量。

**验收：** `generate -p "hello" -n 8` 能结束并输出最多 8 个 token；`-n 0`、缺失模型和超长 prompt 的行为定义清楚；相同模型和 greedy 输入结果可复现。

### 阶段 8：可读的调用路径与基础文档

**小目标：** 新手能从 README 亲自构建并理解一次推理经过哪些模块。

- README 写先决条件、PowerShell 构建、生成 Tiny 模型、运行 generate 和 CTest 命令。
- 用一张简短流程图说明 CLI → loader/tokenizer → context → forward → sampler → 输出。
- 写明 Tiny 随机权重不具备真实聊天能力，并解释 `model.json`、`model.bin`、`vocab.json` 各自作用。
- 清理当前阶段不用的 placeholder；不为了目录外观而保留空接口。

**验收：** 在全新 build 目录按 README 操作能成功；CMake/CTest 默认只构建运行当前阶段必需的目标。

### 阶段 9：Prefill/Decode 边界、采样与健壮性

**小目标：** 把单 token MVP 整理为稳定的 CPU 文本生成接口。

- 定义轻量 `MiniBatch` 或等价输入，统一多 token prompt prefill 和单 token decode；保持实现易读。
- 加入 temperature、top-k 和可复现 seed；greedy 作为默认值。
- 规范空 prompt、EOS、top-k 超出词表、temperature 为零/负数和上下文剩余容量策略。
- 增加 `test_sampler.cc` 和 prefill/decode 相关 forward 测试。

**验收：** 默认 greedy 可复现；固定 seed 的随机采样可复现；非法参数清楚报错；prefill 一批 token 与逐 token 输入的最终 logits 在容差内一致。

### 阶段 10：终端多轮聊天

**小目标：** 一次启动后可以反复输入 prompt，直到输入退出命令。

- 增加 `ChatMessage`、`ChatSession` 和简单 prompt builder；实现最小 system/user/assistant 模板。
- 增加 `run <model-dir>` 终端循环，至少支持 `/help`、`/clear`、`/exit`。
- 先在每轮对话中重新 prefill 完整历史，确认行为正确；不先做复杂的 KV prefix 复用。
- 添加 `test_chat.cc` 覆盖消息顺序、prompt 拼接和清空历史；CLI smoke 检查退出与一轮交互。

**验收：** 可连续完成两轮输入；对话模板的消息角色顺序正确；clear 后旧消息不再进入 prompt；超出上下文有说明并可继续退出。

### 阶段 11：GGUF 文件 reader

**小目标：** 读取 GGUF header、metadata、tensor 描述和数据偏移，但暂不运行真实模型。

- 定义 GGUF magic/version、metadata 类型、tensor 描述和对齐规则。
- 先支持读取 metadata 与 tensor table；严格检查版本、数量、文件截断、重复键和对齐偏移。
- 实现 `inspect-gguf <file>` 只读命令，打印模型 metadata 和 tensor name/shape/type。
- 只新增 `gguf.h/.cc`、最小的 GGUF fixture 生成脚本或测试内存文件，以及 `test_gguf.cc`。

**验收：** reader 能读取提供的合法 GGUF header；错误 magic、越界 offset、未知不支持版本和截断文件被拒绝；不需要下载大模型就能跑测试。

### 阶段 12：GGUF 模型映射与 F32 权重

**小目标：** 将常见 LLaMA/Qwen2 命名的 F32 GGUF tensor 映射为内部模型结构。

- 从 metadata 构造模型配置；将 embedding、各层 attention/FFN 权重、norm 和 lm_head 映射到阶段 3 的模型类型。
- 第一小步只支持 F32 tensor、单一已知 architecture 和完整权重映射；对缺失/额外不兼容 tensor 明确报错。
- 加 `inspect <model-dir>` 或模型检查输出，展示配置和权重概况。
- 添加 `gguf_loader` fixture 和 `test_gguf_loader.cc`；选择许可明确、体积小的模型做手工 smoke 验证。

**验收：** 合成 F32 GGUF 能映射并通过前向；模型 architecture/shape 不匹配时失败；Tiny JSON loader 保持可运行。

### 阶段 13：GGUF Tokenizer 与真实模型 smoke

**小目标：** 读取 GGUF 词表和 merges，能够使用小型真实模型生成 token。

- 按模型 metadata 解析 vocab、merges、BOS/EOS/UNK 和特殊 token；实现 BPE encode/decode 与 byte fallback。
- 处理 GGUF chat template；若模板不在第一版支持范围，先明确限制并允许外部模板参数。
- 先用 `inspect-gguf`、`generate -n 1` 验证 loader 和 tokenizer，不以回答质量做正确性判断。
- 只保留必要的 tokenizer fixture 和一份真实模型 smoke 命令；大模型权重不作为必须提交的文件。

**验收：** 对选定 GGUF，prompt tokenization、单步前向、EOS/BOS 处理和 decode 均正常；无法支持的 tokenizer metadata 给出可定位错误。

### 阶段 14：量化 Tensor 与 Q8_0

**小目标：** 在内部统一量化权重存储，并完成第一种 block quantized linear。

- 定义 dtype、量化 block 布局和反量化/点积接口；embedding/norm 等暂保留 F32。
- 首先实现 Q8_0 的 block 编码、解码和 Linear；比较与 F32 的最大/平均误差。
- GGUF loader 识别 Q8_0 tensor，并将其映射到量化存储；不一次性支持所有 GGUF 类型。
- 添加 `test_quant.cc` 覆盖 block 边界、scale、反量化和线性层误差。

**验收：** Q8_0 量化/反量化 shape 一致且数值有限；Q8_0 linear 与 F32 误差在项目定义的容差内；F32 Tiny 路径回归通过。

### 阶段 15：Q4_0 与 Q4_1

**小目标：** 扩展两种常见低位量化格式并加载相应 GGUF 权重。

- 在 Q8_0 接口基础上实现 Q4_0，再实现 Q4_1；分别记录 scale/zero point 和 block 编码规则。
- loader 按 tensor type 选择对应存储和 CPU Linear；其他 K-quants/IQ 格式明确显示不支持。
- 对每种类型测试最小/最大值、奇偶索引、block 尾部和数值误差。

**验收：** 支持类型可加载并生成至少一个 token；不支持类型有明确报错，不会误当 F32 读取；量化数值测试通过。

### 阶段 16：CPU 多线程与 dispatch

**小目标：** 在不改变结果的前提下并行加速矩阵乘。

- 先用固定行分块实现线程池或按行并行；只有实测性能有意义时再加入线程池抽象。
- 给 `MatMul` 增加朴素/并行实现选择和线程数配置；小矩阵允许回退朴素实现。
- 明确线程生命周期、任务异常传播和多次调用安全性；不先引入复杂 SIMD dispatch。
- 添加 `test_threaded_matmul.cc` 对比 1 线程与多线程数值结果，并检查边界行。

**验收：** 多线程结果与基础实现差异在浮点容差内；连续多次运行不死锁；大矩阵 benchmark 可观察到合理性能差异。若无性能收益，保留朴素路径作为默认。

### 阶段 17：Benchmark 与请求诊断

**小目标：** 可测量 prefill/decode 耗时并定位失败阶段。

- 增加 `bench` 命令，报告模型、prompt token 数、生成 token 数、prefill 时间、decode 时间和 token/s。
- 计时使用 `std::chrono::steady_clock`；不把加载/编译时间混入推理时间。
- 基础错误输出包含阶段和异常内容；仅在确有调试需求时增加 logits top-k 或 tensor shape dump。
- 可选增加一个小型 request trace 结构；不实现复杂遥测和持久化报告。

**验收：** 固定 prompt 能重复运行并打印合理的非负计时；生成数为 0、空 prompt 和加载失败仍能安全报告。

### 阶段 18：CUDA 构建开关与 runtime

**小目标：** CPU 构建继续独立；有 CUDA Toolkit 时能发现设备并管理显存。

- 增加 `MINI_LLAMA_CUDA` CMake 选项，默认 `OFF`；仅开启时启用 CUDA 语言、查找 Toolkit 和编译 `.cu`。
- 封装设备枚举、选择、错误检查、显存 RAII buffer 和 host/device 拷贝。
- CPU-only 下相关 backend 报“未启用 CUDA”并返回明确错误，不要求安装 CUDA 头文件或运行库。
- 增加 CUDA runtime 的 CPU-only 测试和条件式 GPU smoke 测试。

**验收：** 默认 configure/build 不探测或链接 CUDA；启用 CUDA 时能报告设备/无设备；RAII buffer 在分配/释放和异常路径均安全。

### 阶段 19：CUDA Tensor、Linear 与量化 Linear

**小目标：** 先把少数可独立核对的算子跑到 GPU。

- 增加 CUDA Tensor 的 shape 和设备内存所有权；上传/下载显式发生并可统计。
- 先实现 F32 Linear/cuBLAS，再实现 Q8_0，最后评估 Q4_0/Q4_1 CUDA kernel。
- 设备 id、shape 和数据类型不匹配时立即失败；未支持权重可明确 fallback 到 CPU。
- 添加 CUDA/CPU 结果对照测试，按有无 GPU 分层执行。

**验收：** GPU 结果与 CPU 参考误差符合容差；CPU-only build/test 不依赖 GPU；无 GPU 时 CUDA smoke 被清楚标记为跳过或 unavailable。

### 阶段 20：CUDA activation、KV cache 与 attention

**小目标：** 将 decode 主链路中主要算子和 KV cache 放在 GPU。

- 按顺序实现 GPU RMSNorm、RoPE、SiLU/elementwise、softmax、GPU KV cache 和单 token causal attention。
- 先验证单算子，再连接一层 forward，最后连接完整 Transformer forward；每个阶段与 CPU logits 对照。
- 追踪 host/device 拷贝，避免每一层都把中间激活传回 CPU；采样阶段可以保留 logits 回传 CPU。
- GPU KV cache 清零、重置和上下文边界都要有测试。

**验收：** CUDA 单步 logits 与 CPU 结果在容差内；多步生成不越界；GPU 上下文重置后不读取上一请求残留值；没有 GPU 的环境仍能运行 CPU 版本。

### 阶段 21：最终功能对照与精简

**小目标：** 对照参考项目确认目标功能覆盖，并把新手项目仍不需要的部分排除在默认路径外。

- 建立功能对照表：构建方式、CLI 命令、Tiny manifest、GGUF reader/loader/tokenizer、F32/Q8_0/Q4_0/Q4_1、聊天、采样、KV cache、CPU 优化、benchmark、可选 CUDA。
- 每项记录支持状态、限制、对应测试和最小运行方式；明确参考项目也不支持的 tensor 类型仍不支持。
- 默认 README 只呈现 CPU 入门路径；高级格式、量化和 CUDA 分章说明。
- 删除重复测试、未用 Python 工具、过期接口和未引用空文件；保留最短的回归路径和能复现 bug 的测试。
- 最终从干净 checkout 验证 CPU build/test；另在 CUDA 机器验证 CUDA configure/build/test。

**验收：** 功能表中的承诺都有一条可运行验证；README 从零构建成功；CPU-only 用户无需安装 CUDA；高级功能不能悄悄改变 Tiny CPU 的基线结果。

## 4. 最小测试与脚本保留规则

测试文件随功能进入，不一次性复制参考项目的测试目录：

| 进入阶段 | 只新增的必要验证 |
|---|---|
| Tensor/算子 | 自定义 C++ runner、`test_tensor.cc`、`test_ops.cc` |
| loader/tokenizer/cache/forward/sampler | 每个模块一个对应 C++ 测试文件 |
| CLI/chat | 一份 Python CLI smoke；覆盖正常生成、错误退出和基本聊天命令 |
| GGUF | 一个小型合成 fixture 和 reader/loader/tokenizer 单元测试；不依赖下载模型跑普通 CTest |
| 量化/并行 | 与 F32/单线程参考值对照的 C++ 测试 |
| CUDA | 条件式 GPU 测试；普通 CPU 测试不要求 CUDA 机器 |

Python 脚本只保留模型导出、必要 CLI smoke 和无法用小型 C++ fixture 代替的格式辅助工具。每个测试都要对应一个风险或行为，避免为了覆盖文件数量而复制重复测试。

## 5. 每阶段交付节奏

每个里程碑按以下顺序完成，并在阶段结束时汇报：

1. 用一句话说明本阶段小目标和它依赖的上阶段结果。
2. 先写接口与最小实现，控制新增文件数量。
3. 增加能证明该小目标成立的最小测试或命令。
4. 执行 Windows CMake build 和该阶段测试；记录实际命令与结果。
5. 更新 README 的构建/运行示例和已知限制。
6. 只有本阶段验收通过，才继续下一阶段；如果发现设计问题，先修复并补回归测试。

推荐里程碑顺序：

```text
空项目构建
  → Tensor/算子
  → Tiny 模型和 tokenizer
  → KV cache/Transformer forward
  → greedy generate
  → 采样和多轮聊天
  → GGUF/真实模型
  → 量化
  → CPU 并行与 benchmark
  → 可选 CUDA
  → 功能对照与清理
```

## 6. 早期 Windows 命令约定

阶段 0 之后，README 的基础命令保持以下形式，避免要求新手先掌握多个构建系统：

```powershell
# 在新项目根目录
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure

# 生成教学用 Tiny 模型并运行
python scripts/export_tiny_model.py --seed 42 --output-dir models/tiny
& .\build\Debug\mini-llama.exe generate --model models\tiny -p "hello" -n 8
```

若使用 Ninja 或 MinGW，执行文件路径可能不同；文档以 CMake 多配置生成器的 `build\Debug` 作为 Windows 默认示例，并在实际开始编码时根据所选生成器调整命令。

## 7. 明确不提前做的内容

- 阶段 0–8 不下载或提交大型模型，不实现 GGUF、BPE、量化、CUDA、多线程、Radix Tree 或复杂 telemetry。
- 先不做训练、反向传播、自动微分、服务端 API、多 GPU、流式网络协议、图形界面或生产级模型管理。
- 不因最终目录树列出了文件就提前创建空实现；文件随着对应阶段目标出现。
- 不以随机 Tiny 模型的文本质量验收算法。正确性用 shape、有限 logits、固定输入复现性、CPU/GPU 对照和小型真实模型 smoke 判断。

