---
title: 从 AI 编译器项目理解 AI Infra
date: 2026-07-01
category: tech
category_name: 技术笔记
routes: operator-library,team-collaboration
summary: 记录参与 Triton 共性前端和硬件后端适配项目后，对 AI 推理系统、基础算子、IR、Pass、Adapter 和 JIT 的阶段性理解。
---

这段时间我参与了一个 AI 编译器相关项目，核心是围绕 Triton 构建一个共性前端，并适配不同硬件后端。

这个项目让我意识到，AI Infra 并不只是训练模型或者调用模型 API。它还包括模型推理背后的编译器、算子库、运行时、硬件后端、性能基准、CI/CD，以及一整套工程化保障。

这篇文章先记录我目前对整体框架的阶段性理解。后续我会再围绕每个部分展开，比如 Triton kernel、典型算子、IR 转换、后端适配、profiling 和 CI 流程。

## 1. AI Infra 在模型推理中的位置

大模型推理表面上是在执行一个模型，但从系统角度看，它其实是大量算子的组合执行。矩阵乘法、加法、softmax、layernorm 等算子会反复出现在 Transformer 结构中。

如果把应用层、模型资产、服务框架、编译运行时和硬件资源放在一张图里，AI Infra 的核心链路大致处在下面这个位置：

![AI Infra 在模型推理系统中的位置](/assets/images/ai-infra-position.svg "AI Infra 是连接业务应用、模型资产、推理服务、编译运行时和硬件资源的系统工程层。")

也就是说，AI Infra 不是模型本身，也不是某一块具体硬件，而是让模型可以被部署、服务、优化、观测和维护的系统工程层。

这一部分我们重点研究的是编译优化与算子运行层。高性能算子通常有两类实现思路：一种是直接编写 CUDA 高性能代码，从算法和硬件特性出发，手动设计访存方式、并行粒度、线程映射和数据布局；另一种是通过 AI 编译器自动生成、优化或转换算子代码，把优化逻辑沉淀到编译流程、IR 表达和后端适配中。

这两种方式各有优势。手写 CUDA 的优势在于控制力强，开发者可以针对具体算子、具体硬件和具体数据规模做非常细致的优化，往往更容易逼近单个场景下的性能上限；但它的缺点也很明显：开发门槛高、调试成本高、可移植性弱，一旦算子形态、输入规模或硬件平台发生变化，很多优化都需要重新设计。

AI 编译器的优势则在于复用性和自动化。它希望把常见的优化策略抽象到编译器系统中，通过 IR、调度规则和后端代码生成，让更多算子和更多硬件共享同一套优化基础设施。这种方式可以降低高性能算子开发的门槛，也更适合面对模型结构快速变化、硬件平台多样化的 AI Infra 场景。不过，编译器生成代码也并不意味着一定能超过手写 CUDA：它依赖编译器对算子语义、硬件特性和调度空间的建模能力，在极致性能、特殊算子或复杂边界条件下，仍然可能需要人工介入和针对性优化。

因此，AI Infra 与传统 HPC 的交叉点并不是简单地用编译器替代手写优化，而是把 HPC 中对计算模式和硬件细节的理解，进一步系统化地沉淀到编译器基础设施中。手写 CUDA 更像是面向具体问题的精细优化，AI 编译器则更强调把这些优化经验变成可复用、可迁移、可扩展的系统能力。

## 2. 从 Kernel 算子理解推理优化

在 AI Infra 和 AI 编译器中，算子是连接模型表达、编译器优化和硬件执行的核心单位。深度学习框架中通常使用 operator 描述高层计算，例如 `torch.add`、`torch.mm`、`torch.softmax`；而在编译器和硬件执行层面，更关注 kernel，也就是某个算子在具体硬件上的执行实现。

同一个 operator 在不同 shape、dtype、硬件后端和调度策略下，可能会对应不同的 kernel。推理优化通常不是只看完整模型，而是需要拆解模型中高频出现的基础算子，分析它们的计算模式、访存模式、数值稳定性和可融合性。

常见推理算子可以大致分为几类：

- Elementwise：代表算子包括 add、bias、activation，优化重点是减少访存和算子融合。
- Matrix Compute：代表算子包括 GEMM、QK^T、PV，优化重点是 tiling、矩阵计算单元和数据复用。
- Reduction：代表算子包括 softmax、layernorm、RMSNorm，优化重点是并行归约和数值稳定。
- Attention-specific：代表算子包括 RoPE、FlashAttention、KV cache，优化重点是分块、融合和减少 HBM 访问。
- Generation：代表算子包括 top-k、sampling，优化重点是降低逐 token 延迟。

### 2.1 Add：最基础的 Elementwise 算子

Add 是逐元素加法，最简单的形式是：

```text
C[i] = A[i] + B[i]
```

如果涉及 broadcasting，也可能是：

```text
C[i, j] = A[i, j] + B[j]
```

Add 在 Transformer 推理中非常常见，例如：

- residual connection：`x + attention(x)`
- bias add：`matmul(x, w) + bias`
- attention mask：`attention_score + attention_mask`
- embedding add：`token_embedding + position_embedding`

Add 的计算量很小，通常不是 compute-bound，而是 memory-bound。也就是说，性能瓶颈主要来自内存读写，而不是加法本身。一个普通 add kernel 至少需要读取两个输入 tensor，再写回一个输出 tensor。

常见优化方向包括：

- 保证输入输出连续，提升内存访问效率
- 使用向量化 load/store，一次处理多个元素
- 优化 broadcasting，避免重复加载小张量
- 与 bias、activation、residual 等 elementwise 操作融合
- 减少中间 tensor 写回

Add 虽然数学上简单，但在编译器项目中非常适合作为最小闭环测试。它可以验证从高层算子入口到 IR 生成、Pass pipeline、后端 lowering 和运行时执行的基础链路是否打通。

### 2.2 GEMM / Matmul：推理中的核心计算密集型算子

GEMM 是通用矩阵乘法，形式为：

```text
C = A x B
```

展开后是：

```text
C[i, j] = sum(A[i, k] * B[k, j])
```

其中：

- A: `[M, K]`
- B: `[K, N]`
- C: `[M, N]`

GEMM 是 Transformer 推理中最重要的计算之一。大量线性层本质上都是矩阵乘法，例如：

- `Q = XWq`
- `K = XWk`
- `V = XWv`
- `O = Attention(Q, K, V)Wo`
- MLP 中的 up projection / down projection
- hidden states 到 vocabulary logits 的投影

Attention 中的 `QK^T` 和 `softmax(QK^T)V` 本质上也可以视为矩阵乘法。

GEMM 通常是 compute-bound 算子，尤其在矩阵规模较大时，性能主要取决于硬件矩阵计算单元的利用率。常见优化方向包括：

- tiling：将大矩阵切成小块
- blocking：让 tile 数据尽量驻留在 cache / shared memory / local memory 中
- data reuse：提高 A、B tile 的复用率
- 使用 Tensor Core / NPU / TPU 矩阵单元
- fp16 / bf16 输入，fp32 accumulation
- 调整 layout，减少转置和非连续访问
- matmul + bias + activation 融合
- int8 / int4 量化 GEMM

GEMM 是检验编译器和后端能力的重要算子。它不仅要求正确 lowering 矩阵乘法，还要求能生成适合硬件矩阵单元的高效代码。

### 2.3 Softmax：Attention 中的数值稳定归一化

Softmax 用于将一组分数转换为概率分布：

```text
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

实际实现中通常使用数值稳定版本：

```text
m = max(x)
softmax(x_i) = exp(x_i - m) / sum(exp(x_j - m))
```

减去最大值不会改变 softmax 的数学结果，但可以避免 `exp(x)` 溢出。

Softmax 最典型的使用位置是 attention：

```text
score = QK^T / sqrt(d)
prob = softmax(score)
output = prob x V
```

Softmax 的计算模式通常包含：

- max reduction
- subtract max
- exp
- sum reduction
- divide by sum

因此它既包含归约，也包含 elementwise 操作。优化方向包括：

- 高效并行归约 max 和 sum
- 减少中间结果写回全局内存
- 保证数值稳定性
- 优化 attention mask 处理
- 使用近似 exp 或硬件 exp 指令
- 与 attention 融合，减少 score / prob 中间矩阵写回

FlashAttention 是 softmax 相关优化的典型代表。其核心思想是分块计算 attention，并使用 online softmax 维护局部和全局归一化信息，从而避免显式保存完整 attention score 矩阵，大幅减少 HBM 访问。

### 2.4 LayerNorm：Transformer 中的高频归一化算子

LayerNorm 的公式为：

```text
mean = sum(x) / N
var = sum((x - mean)^2) / N
y = (x - mean) / sqrt(var + eps) * gamma + beta
```

其中：

- gamma：缩放参数
- beta：平移参数
- eps：防止除零

LayerNorm 在 Transformer 中非常常见，例如：

- layernorm -> attention
- residual add -> layernorm
- layernorm -> MLP

LayerNorm 的计算模式是 reduction + elementwise。它需要先计算均值和方差，再对每个元素做归一化、缩放和平移。相比 GEMM，它的计算密度较低，更容易受到访存、归约效率和 kernel launch 开销影响。

常见优化方向包括：

- 高效计算 mean 和 variance
- 使用 fp32 accumulation 保证统计稳定性
- 向量化 hidden dimension 的读取和写回
- 减少中间 buffer
- residual add + layernorm 融合
- bias + layernorm 融合
- 针对固定 hidden size 做特化 kernel

LayerNorm 代表了一类不同于 GEMM 的优化问题：它不是为了吃满矩阵计算单元，而是要减少访存、降低归约开销，并提升小规模并行效率。

### 2.5 RMSNorm：LayerNorm 的常见替代

RMSNorm 是 Root Mean Square Normalization。它不计算均值，只基于均方根做归一化：

```text
rms = sqrt(mean(x^2) + eps)
y = x / rms * weight
```

相比 LayerNorm，RMSNorm 少了均值计算，结构更简单。许多 LLM 使用 RMSNorm，例如 LLaMA 系列。

常见优化方向包括：

- 高效计算 `sum(x^2)`
- 使用 fp32 accumulation
- 向量化 hidden dimension
- residual add + RMSNorm 融合
- 针对固定 hidden size 特化

RMSNorm 是推理系统中很值得关注的归一化算子，因为它在现代 LLM 中出现频率很高。

### 2.6 Activation：GELU / SiLU / ReLU

Activation 用于引入非线性。常见激活函数包括：

```text
ReLU(x) = max(0, x)
SiLU(x) = x * sigmoid(x)
GELU(x) ~= x * Phi(x)
```

它们通常出现在 MLP / FFN 中：

```text
Linear -> Activation -> Linear
```

在 LLaMA 等模型中，还常见 SwiGLU 结构，其核心包括 SiLU 相关计算。

Activation 通常是 elementwise 算子，优化重点包括：

- 与 bias add 融合
- 与 GEMM epilogue 融合
- 使用近似公式降低 exp / tanh 成本
- 向量化 elementwise 计算
- 减少中间 tensor 写回

Activation 本身计算不一定重，但出现频率高，且非常适合与前后的线性层或 elementwise 操作融合。

### 2.7 RoPE：Rotary Position Embedding

RoPE 用于向 Query 和 Key 注入位置信息。它通过对向量的偶数维和奇数维做旋转变换实现：

```text
q_even, q_odd -> rotate(q, sin, cos)
k_even, k_odd -> rotate(k, sin, cos)
```

RoPE 通常出现在 attention 计算之前：

```text
Q, K = apply_rope(Q, K)
score = QK^T
```

常见优化方向包括：

- 预计算 sin / cos
- 向量化偶数维和奇数维处理
- 与 Q/K projection 融合
- 与 attention kernel 融合
- 减少 Q/K 中间结果写回

RoPE 是 LLM 推理中很典型的 attention-specific 算子。

### 2.8 Attention / FlashAttention

标准 attention 计算流程是：

```text
score = QK^T / sqrt(d)
prob = softmax(score)
out = prob x V
```

其中包含 GEMM、scale、mask、softmax 和 GEMM。如果直接实现，可能会显式生成完整的 score 矩阵和 prob 矩阵，造成大量内存读写。

FlashAttention 的核心优化思路是：

- 分块计算 `QK^T`
- 在线维护 softmax 的 max 和 sum
- 不显式保存完整 attention score
- 直接累积输出

它的优化重点包括：

- block tiling
- online softmax
- 减少 HBM 访问
- causal mask 融合
- KV cache 友好访问
- 多 head / grouped-query attention 优化

Attention 是推理优化中最核心的组合型算子之一。

### 2.9 Quant / Dequant

量化用于降低数据精度，从而减少存储和带宽压力：

```text
fp16 / bf16 -> int8 / int4
```

反量化则将低精度数据恢复到计算所需格式：

```text
int8 / int4 -> fp16 / fp32
```

在 LLM 推理中，常见量化方式包括：

- weight-only int8 / int4
- activation quantization
- KV cache quantization

优化方向包括：

- dequant 与 GEMM 融合
- per-channel / per-group scale 优化
- 减少 scale / zero point 加载开销
- 使用低精度矩阵指令
- 平衡精度损失和吞吐提升

量化相关算子对推理部署非常重要，尤其在显存容量和带宽受限时。

### 2.10 Top-k / Sampling

生成式推理最后需要从 logits 中选择下一个 token。常见流程包括：

```text
logits
 -> temperature
 -> top-k / top-p
 -> softmax
 -> sampling
```

这些算子出现在 decoding 阶段，每生成一个 token 都会执行。

优化方向包括：

- top-k 选择算法优化
- 减少 vocabulary 维度扫描成本
- softmax 与 sampling 融合
- batch decoding 优化
- 降低 CPU/GPU 同步开销

虽然 sampling 不是 Transformer block 内部最重的计算，但它会影响逐 token 生成延迟。

### 2.11 小结

从 kernel 算子的角度看，推理优化可以理解为不同计算模式的组合优化：

- elementwise：关注访存和融合
- GEMM：关注矩阵单元和数据复用
- reduction：关注并行归约和数值稳定
- attention：关注分块、融合和 KV cache
- generation：关注逐 token 延迟

这些算子共同构成了模型推理的底层执行路径。理解它们的数学语义、使用位置和优化方式，是理解 AI Infra、编译器优化和后端性能调优的基础。

## 3. 我对 Triton、FlagGems 和后端的理解

在这个项目中，Triton 更像是一个面向 AI 算子的编程和编译基础设施。开发者可以用类似 Python 的方式写 kernel，然后由 Triton 编译到更底层的表示。

FlagGems 可以理解为一个算子库。它提供了很多常见算子的 Triton 实现。对于我们做性能基准来说，add、GEMM、softmax、layernorm 这些 kernel 可以从 FlagGems 的算子入口触发，而不是临时手写一份和真实算子库无关的 kernel。

后端则是面向具体硬件的部分。不同厂商硬件的执行模型、内存结构、指令能力和编译工具链都不一样，因此需要 out-of-tree backend 来完成从共性前端到具体硬件的接入。

我现在对这几个部分的理解是：

- 前端解决如何统一表达和转换
- 后端解决如何落到具体硬件执行
- FlagGems 提供典型算子入口
- Triton / Anchor 提供编译和 IR 转换基础设施

这里的关键不是某一个 kernel 能不能跑，而是能不能形成一条可复用、可扩展、可测试的链路。只有链路稳定，后续才谈得上系统性地接入更多算子、更多硬件和更多优化策略。

## 4. 我对 AI 编译器流程的理解

项目中经常出现几个关键词：

- IR
- Pass
- Adapter
- JIT
- Profiling

IR 是中间表示。它不是最终机器码，也不是最上层 Python 代码，而是编译器内部用于分析、优化和转换的结构化表示。理解 IR 的价值在于，它把高层算子语义和底层硬件执行之间切开了一层，使编译器可以在中间阶段做系统化处理。

Pass 是编译器中的一个处理阶段。一个 Pass 可能做优化，也可能做 IR 转换。例如从 Triton 相关 IR 转到 Linalg，再转到后端需要的表示。多个 Pass 串起来，就形成了一条 compilation pipeline。

Adapter 是连接不同 IR 或不同工具链的桥梁。在这个项目中，Adapter 负责把前端生成的表示转换到后端可以继续处理的形式。它看起来像胶水层，但实际上决定了前端抽象能否被具体硬件后端正确承接。

JIT 是 Just-In-Time 编译，也就是运行时触发编译。它和传统 C++ 那种先编译、再运行的流程不完全一样。Triton kernel 往往是在第一次调用时触发编译，因此编译时间和首次执行时间会耦合在一起。

Profiling 则是理解性能表现的入口。只知道一个 kernel 能跑还不够，还要知道它的耗时、吞吐、瓶颈在哪里，以及不同后端之间的差异是否符合预期。对 AI Infra 来说，profiling 不只是调参工具，也是验证编译器和后端质量的重要手段。

## 5. 当前阶段的收获

这个项目对我最大的帮助，是把我对高性能计算的理解从单点 kernel 扩展到了完整系统链路。

以前我更关注一个 kernel 内部怎么写得更快，比如访存合并、数据复用、线程组织和计算访存比。现在我会进一步追问：这个 kernel 从哪里来，经过了哪些 IR，哪些 Pass 改写了它，后端如何接住它，profiling 如何证明它真的变快了，以及 CI 如何保证这些能力不会在后续迭代中退化。

这让我对 AI Infra 有了更具体的认识。它不是一个抽象名词，而是一套让模型能够稳定、高效、可迁移地跑在真实硬件上的工程系统。编译器、算子库、运行时和硬件后端都只是其中的一部分，真正重要的是这些部分之间如何形成闭环。
