# 多 Token 预测（Multi-Token Prediction，MTP）

> 从 GPT-2 到 Llama 3，每个自回归 LLM 都按位置训练一个损失：预测下一个 token。DeepSeek-V3 给每个位置加了第二个损失：再预测再下一个 token。在 671B 主模型上多出来的 14B 参数，通过梯度流蒸馏回主模型；训练好的 MTP 头在推理时被改造成投机解码（speculative decoding）的草稿器（drafter），接受率超过 80%。1.8× 生成吞吐几乎白送。本课按 DeepSeek 技术报告构建顺序 MTP 模块，计算损失与共享头的参数布局，并解释为什么 MTP 保住了因果链，而 Gloeckle 等人最初的并行 MTP 把它打断了。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 04 (pre-training a mini GPT), Phase 10 · 15 (speculative decoding)
**Time:** ~60 minutes

## 学习目标（Learning Objectives）

- 陈述 MTP 训练目标，并推导跨预测深度的联合损失
- 解释 Gloeckle 等人的并行 MTP 头（2024）与 DeepSeek-V3 顺序 MTP 模块的差异，以及为什么顺序设计保住因果链
- 计算给预训练运行添加 MTP 模块的参数与内存开销
- 从零实现一个 MTP 模块：共享嵌入、每深度 transformer 块、投影，以及共享输出头

## 问题（The Problem）

下一 token 预测是标准 LLM 训练目标。每个隐状态都被监督去预测恰好一件事：紧跟着的那个 token。这个信号出奇地弱。序列里的大部分信息延伸到一个 token 之外——结构、连贯、事实性、算术流程。模型必须通过在数万亿 token 上累积大量单 token 信号来学习这些。

MTP 问：如果每个隐状态都被监督去同时预测多个未来 token 会怎样？Gloeckle 等人（Meta，2024）证明这有帮助。他们的实现是在主干上放几个独立输出头，各自预测不同偏移。并行、简单，但这些头看到的是同一个隐状态，没有分层精炼——预测也不按因果链式衔接，因此不能用于投机解码。

DeepSeek-V3（2024 年 12 月）把 MTP 重设计成顺序模块，在每个预测深度都保住因果链。模型从 `h_i^(0)` 预测 `t+1`，再从一个把 `h_i^(0)` 与 `E(t+1)` 嵌入组合起来的新隐状态 `h_i^(1)` 预测 `t+2`，依此类推。每个深度是自己的小型 transformer 块。共享嵌入和共享输出头让参数开销保持温和。在 DeepSeek-V3 的规模上，MTP 模块在 671B 主模型权重之上额外 14B 参数。这 2% 的开销买来了更密的训练信号，以及推理时现成的投机解码草稿。

本课从零构建单个 MTP 模块和 D 深度损失。数学干净。实现大约 150 行。

## 概念（The Concept）

### 顺序 MTP 配方（The sequential MTP recipe）

DeepSeek-V3 在主模型之上加 `D` 个 MTP 模块。每个模块 `k`（`k = 1..D`）预测深度 `k` 的 token——即给定到位置 `i` 的前缀，预测 `t_{i+k}`。

模块 `k` 包含：

- 带自己注意力和 MLP 的 transformer 块 `T_k`。
- 投影矩阵 `M_k`，把上一深度隐状态与下一深度真值 token 的嵌入组合起来。
- 共享嵌入 `E`（与主模型相同）。
- 共享输出头 `Out`（与主模型相同）。

训练时，对到位置 `i` 的前缀，每深度隐状态是：

```
h_i^(0) = main model backbone at position i
h_i^(k) = T_k( M_k * concat(RMSNorm(h_i^(k-1)), RMSNorm(E(t_{i+k}))) )   for k >= 1
```

每深度预测是：

```
logits_{i+k} = Out(h_i^(k-1))   for k = 1..D
```

每深度损失是相对真值 `t_{i+k}` 的交叉熵：

```
L_k = CE(logits_{i+k}, t_{i+k})
```

跨深度的联合损失：

```
L_MTP = (lambda / D) * sum_{k=1..D} L_k
```

`lambda` 是一个较小的加权因子——DeepSeek-V3 在训练前 10% 用 0.3，之后用 0.1。总训练损失是 `L_main + L_MTP`。

### 为什么用顺序而不是并行（Why sequential, not parallel）

Gloeckle 最初的并行 MTP 有 D 个输出头，各自直接作用在 `h_i^(0)` 上。每个头从同一个主干隐状态预测 `t_{i+k}`。这样能训，但预测彼此不条件化。你不能用 `head_1` 的输出去帮 `head_2`——这些头并行开火。

DeepSeek-V3 的顺序设计用 `h_i^(k-1)` 加上真实下一 token 嵌入 `E(t_{i+k})` 来构建 `h_i^(k)`。这保住了因果链：要预测 `t_{i+k+1}`，深度 `k+1` 的模块看到 `t_{i+k}` 处有什么。这在结构上等同于自回归解码器消费自己的输出——因此 MTP 模块可以直接当投机解码草稿器用。

推理时：把 `h_i^(k-1)` 和草稿出的 `t_{i+k}` 送进模块 `k+1`，得到 `t_{i+k+1}` 的预测。重复。这恰好是 EAGLE 风格的草稿，用训练好的 MTP 模块当草稿网络。DeepSeek-V3 报告第一个 MTP 模块接受率超过 80%，加速约 1.8×。

### 参数核算（Parameter accounting）

对隐层为 `h`、词表为 `V` 的模型：

- 主模型：数十亿参数，外加一个大小为 `V * h` 的输出头。
- 共享输出头：复用主模型的头。没有额外参数。
- 共享嵌入：复用主模型的嵌入。没有额外参数。
- 每个 MTP 模块：
  - 投影 `M_k`：`(2h) * h = 2h^2`。
  - Transformer 块 `T_k`：注意力（MHA 为 `4h^2`）加上 MLP（SwiGLU、比例 8/3 时通常 `8h^2`）。每块大约 `12h^2`。

每模块额外总计：`~14h^2`。对 DeepSeek-V3 的 `h = 7168`、D = 1 个模块：纸面上 `~14 * 7168^2 = ~720M` 参数。DeepSeek-V3 报告 14B——差额主要是 MTP 模块里也有 MoE 专家层。

### 投机解码的回报（The speculative-decoding payoff）

预训练期间，MTP 模块大约让训练变慢 10%（更多前向计算，额外损失）。回报是双重的：

1. 更密的训练信号。每个隐状态看到 D+1 个监督目标。DeepSeek-V3 消融中，MMLU、GSM8K、MATH、HumanEval 上测到稳定的几个百分点提升。

2. 推理时免费的投机解码草稿。MTP 模块已经被训练去预测接下来几个 token。改造成草稿网络后，接受率超过 80%。在这个水平上，N=3 或 N=5 的投机解码给出 1.8× 吞吐。10% 的训练时间成本，第一次跑推理就回本。

### 与 EAGLE 的关系（Relation to EAGLE）

EAGLE 在预训练之后单独训练一个小草稿模型。MTP 把草稿烤进预训练。两条路线在相近的接受率上会合，但流水线不同：

| 维度 | EAGLE-3 | MTP（DeepSeek-V3） |
|-----------|---------|------------------|
| 何时训练 | 预训练之后 | 预训练期间 |
| 与已有权重向后兼容 | 是 | 否（需要重新训练） |
| 草稿参数 | 1–2 个 transformer 层 | 1 个 transformer 块 + 投影 |
| 接受率 | 0.88–0.92 | 深度 1 上 0.80+ |
| 加速之外的收益 | 仅投机解码 | 更密的训练信号 + 加速 |

```figure
multi-token-predict
```

## 动手构建（Build It）

`code/main.py` 端到端构建单个 MTP 模块：共享嵌入、投影、transformer 块、共享输出头。然后在短合成序列上计算每深度交叉熵损失，并按组件打印参数量。玩具词表 32 个 token，数字更好读。

### 第 1 步：共享嵌入表（Step 1: shared embedding table）

一张 `vocab_size x hidden` 表，主模型和每个深度的每个 MTP 模块都用。不是第二份拷贝——就是同一个张量。

### 第 2 步：每深度组合（Step 2: the per-depth combination）

```python
def combine(prev_hidden, next_token_embed, M_k):
    # concat along feature dim, then project down to hidden
    concat = rms_norm(prev_hidden) + rms_norm(next_token_embed)  # vector addition stand-in
    projected = matvec(M_k, concat)
    return projected
```

真正的 DeepSeek-V3 把两个经 RMSNorm 的向量拼接成 `[2h]`，再用 `h x 2h` 矩阵投影。玩具版用向量加法以保持标准库简洁。

### 第 3 步：深度 k 的 transformer 块（Step 3: the transformer block at depth k）

自注意力加 MLP。玩具里用一层线性注意力块和 SwiGLU MLP，结构可见，又不需要 numpy。

### 第 4 步：共享输出头（Step 4: the shared output head）

复用主模型的输出投影。词表上的 logits。

### 第 5 步：每深度损失（Step 5: per-depth loss）

softmax(logits) 相对偏移 `k` 处真值 token 的交叉熵。用 `lambda / D` 缩放因子跨深度聚合。

### 第 6 步：参数核算（Step 6: parameter accounting）

打印总参数量、共享（嵌入、头）参数量，以及每模块额外参数量。展示 MTP 额外部分相对主模型大小的比例。

## 使用它（Use It）

MTP 已集成进 DeepSeek-V3（2024 年 12 月）和 DeepSeek-R1 系列。推理时：

- DeepSeek 自己的服务栈开箱即用地把 MTP 模块当投机解码器消费。
- 截至 2026 年 4 月，vLLM 和 SGLang 已有 DeepSeek-V3 MTP 的集成路径。
- AMD 的 ROCm SGLang 教程给出具体的 MTP 投机解码配置，在 V3 检查点上测到 1.8× 加速。

何时在新的预训练运行中用 MTP：

- 你控制完整预训练流水线，想把更密的训练信号存进账户。
- 你知道会大规模服务该模型，想白送投机解码。
- 隐层大小至少 4096。在 1B 规模上，开销伤得比收益帮得更多。

何时不用：

- 微调已有的预训练稠密模型。MTP 模块没有被训练。
- 研究模型，你想要干净基线做对比。MTP 会改架构。

## 交付（Ship It）

本课产出 `outputs/skill-mtp-planner.md`。给定预训练运行规格（模型规模、数据、算力），它返回集成 MTP 的计划：深度数 D、`lambda` 日程、内存开销，以及推理时投机解码的接线。

## 练习（Exercises）

1. 运行 `code/main.py`。展示合成信号变强时每深度损失单调下降。把合成改成固定模式，验证深度 1 和深度 2 损失都收敛。

2. 计算稠密 70B 模型（隐层 8192，80 层）加 D=1 MTP 模块的参数开销。与 DeepSeek-V3 报告的 14B 开销比较。解释为什么 DeepSeek 的数字更高：MTP transformer 块继承同样的 MoE 结构，抬高了每模块参数量。

3. 在玩具里实现 D=2：加第二个 MTP 模块，吃 `h^(1)` 并预测 `t_{i+2}`。验证联合损失和参数核算匹配 DeepSeek 论文方程 19–21。

4. 把玩具改成并行 MTP（Gloeckle 风格）：在主隐状态上加 D 个输出头，各自预测不同偏移。在同一合成信号上比较每深度损失与顺序版本。顺序版本对 k > 1 的深度 k 损失应更低，因为它条件化于中间预测。

5. 把训练好的 MTP 模块当 EAGLE 风格草稿：推理时调用模块 k 提出 `t_{i+k}`。在留出序列上测量这些草稿 token 相对主模型预测的接受率。玩具上若达到 50%+，你就复现了 MTP 当草稿的经验性质。

## 关键术语（Key Terms）

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| MTP 模块 | 「额外损失块」 | 一个小型 transformer 块加投影，预测主模型前方 `k` 个位置的 token |
| 预测深度 | 「哪个偏移」 | 整数 `k`，使得模块 `k` 从到位置 `i` 的前缀预测 `t_{i+k}` |
| 并行 MTP | 「Gloeckle 风格」 | 同一主干隐状态上的 D 个独立头，没有条件链 |
| 顺序 MTP | 「DeepSeek-V3 风格」 | 每个模块条件化于上一深度隐状态加下一 token 嵌入；保住因果链 |
| 共享输出头 | 「复用主头」 | MTP 模块调用主模型的 LM 头，而不是单独的输出投影 |
| 共享嵌入 | 「复用主表」 | 到处用同一张词表嵌入表；没有重复参数 |
| 投影矩阵 M_k | 「组合隐状态 + 下一 token」 | 一个 `h x 2h` 线性层，把上一隐状态和目标 token 嵌入折进下一深度输入 |
| 联合损失 L_MTP | 「平均后的额外损失」 | 每深度交叉熵损失的算术平均，再乘以 `lambda` |
| 深度 1 接受率 | 「MTP 草稿对了多少」 | D=1 MTP 模块的 top-1 预测等于主模型 top-1 的比率；DeepSeek-V3 上 80%+ |
| Lambda 加权 | 「额外损失有多重要」 | 每深度缩放因子；DeepSeek-V3 训练开始时 0.3，后来 0.1 |

## 延伸阅读（Further Reading）

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 完整的顺序 MTP 描述（第 2.2 节），包括联合损失方程和推理时 1.8× 加速
- [Gloeckle et al. — Better & Faster Large Language Models via Multi-token Prediction (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737) — DeepSeek 设计所改进的并行 MTP 基线
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — 总计 685B（671B 主模型 + 14B MTP），部署说明
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — MTP 所嵌入的投机解码框架
- [Li et al. — EAGLE-3 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — EAGLE 的 2025 草稿架构，MTP 的对照物
