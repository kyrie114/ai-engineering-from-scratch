# 差分注意力（Differential Attention, V2）

> Softmax 注意力会把少量概率摊到每一个不匹配的 token 上。超过 10 万个 token 时，这些噪声会累加并把信号淹没。差分 Transformer（Differential Transformer，Ye et al., ICLR 2025）把注意力算成两个 softmax 之差，减掉共享的噪声底。DIFF V2（Microsoft，2026 年 1 月）是生产栈重写：解码延迟对齐基线 Transformer，无需自定义内核，兼容 FlashAttention。本课从 V1 到 V2 端到端走一遍，并带一个可在 stdlib Python 里跑的差分运算玩具实现。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 02 (self-attention), Phase 7 · 15 (attention variants), Phase 10 · 14 (architecture walkthrough)
**Time:** ~60 minutes

## 学习目标（Learning Objectives）

- 精确说明 softmax 注意力为何存在噪声底（noise floor），以及它为何随上下文长度增长
- 推导差分注意力公式，并解释减法如何抵消共享噪声分量同时保留信号
- 走一遍 V1 到 V2 的 diff：什么变快了、什么变简单了、什么更稳定了，以及每一处改动为何是生产预训练所必需
- 用纯 Python 从零实现差分注意力，并在合成的信号加噪声查询上经验验证噪声抵消性质

## 问题（The Problem）

标准 softmax 注意力有一个在规模上会变成运维头痛的数学性质。对查询 `q`，注意力权重是 `softmax(qK^T / sqrt(d))`。Softmax 永远给不出精确的零——每个不匹配 token 都分到一点正质量。那点残余质量就是噪声，并且随上下文长度放大。在 128k token 时，即便每个不匹配 token 只拿到 0.001% 的概率，127,999 个加起来大约贡献总量的 12%。模型必须学会绕开一块随上下文增长的噪声底。

经验上，这表现为注意力头干扰：长上下文 RAG 里的幻觉引用、10 万 token 检索任务上的“迷失在中间”（lost-in-the-middle）失败，以及 32k 之后 needle-in-haystack 基准上微妙的精度下降。差分 Transformer 论文（arXiv:2410.05258，ICLR 2025）测到了差距：DIFF Transformer 相对同规模基线困惑度更低、长上下文精度更高、幻觉更少。

DIFF V1 有三个问题，使它进不了前沿预训练流水线。它的值缓存每个解码步必须加载两次；它需要自定义 CUDA 内核，破坏 FlashAttention 兼容性；它的按头 RMSNorm 在 70B 以上规模的长跑训练中不稳定。DIFF V2（Microsoft unilm 博客，2026 年 1 月 20 日）把三件事都修好了。本课走完两个版本，实现差分算子，并在玩具查询上基准噪声抵消。

## 概念（The Concept）

### Softmax 的噪声底（The noise floor of softmax）

对查询 `q` 和键 `K = [k_1, ..., k_N]`，注意力权重是：

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

没有任何 `w_i` 会是零。若 `k_i` 与 `q` 完全无关，分数 `q . k_i` 也不是 0——它围绕零波动，方差为 `||q||^2 / d`。Softmax 归一化之后，每个无关 token 仍向加权和贡献 `O(1/N)`。无关 token 的总贡献是 `O((N-1)/N) = O(1)`——不是一个小量。

模型想要的是类似硬 top-k：匹配 token 上高权重，别处接近零。Softmax 太光滑，做不到。

### 差分思路（The differential idea）

把每个头的 Q、K 投影拆成两份：Q = (Q_1, Q_2)，K = (K_1, K_2)。计算两张注意力图：

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

输出：

```
DiffAttn = (A_1 - lambda * A_2) V
```

减法抵消两张图共享的噪声分布。若两张图在 12.7 万个无关 token 上都接近均匀权重（随机初始化时确实如此），那些就会抵消。信号——真正相关的少数 token 上的尖峰权重——只有在两张图以相同幅度同时出现时才会抵消，模型训练之后不会如此。

`lambda` 是每个头一个可学习标量，参数化为 `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`。它可以是负的。`lambda_init` 默认是像 0.8 这样的小正数。

### 为何这像带头的噪声抵消（Why this matches headed noise-canceling）

想象两支嘈杂麦克风录同一把声音。两者都拾到说话人加上相关的背景噪声。把一支从另一支里减掉，共享噪声就掉了。声音能留下，是因为两路信号在相位或幅度上差得足以避免完全抵消。按头的 `lambda` 学的正是这种平衡。

### V1 vs V2：diff（V1 vs V2: the diff）

V1 把参数量保持与基线 Transformer 相等。为了每个头得到两个查询，它把头维减半。这牺牲了头的表达力——更痛苦的是——每个头的值缓存也减半。解码每步必须把值缓存加载两次（每个 softmax 分支一次）。结果：尽管参数量匹配，解码比基线更慢。

V2 把查询头数加倍，KV 头数保持不变（参数从上投影借来）。头维与基线相同。减法之后，多出来的维度再投影回去，以匹配基线 Transformer 的 O_W 投影。三件事同时发生：

1. 解码速度匹配基线（KV 缓存只加载一次）。
2. FlashAttention 原样运行（无需自定义内核）。
3. 解码时的算术强度上升（从 HBM 加载的每字节对应更多计算）。

V2 还去掉了 V1 用来稳定减法的按头 RMSNorm。在 70B 级预训练规模上，那个 RMSNorm 会让后期训练失稳。V2 用更简单的初始化方案替代它，无需额外模块就能保持训练稳定。

### 何时该用（When to reach for it）

| 负载 | 收益 |
|----------|---------|
| 长上下文 RAG（64k+） | 更干净的注意力图，更少幻觉引用 |
| Needle-in-haystack 基准 | 32k 之后精度显著提升 |
| 多文档问答 | 更少跨文档干扰 |
| 8k 代码补全 | 边际，不值得改架构 |
| 短对话（< 4k） | 与基线基本无法区分 |

价值随上下文长度增长。4k token 时噪声底足够小，标准注意力就够用。128k 时它在伤害你。

### 如何与其他 2026 旋钮叠放（How it stacks with other 2026 knobs）

| 特性 | 与 DIFF V2 兼容？ |
|---------|------------------------|
| GQA | 是（V2 增加 Q 头，不增加 KV 头） |
| MLA（DeepSeek） | 原则上可以，尚无把二者合在一起的已发表论文 |
| MoE | 是（注意力与 MLP 块独立） |
| RoPE | 是（不变） |
| YaRN / 长上下文缩放 | 是（正是 DIFF 帮助最大的地方） |
| FlashAttention | V2 可以（V1 不行） |
| 投机解码 | 是（注意力改动对投机解码循环不可见） |

```figure
differential-attention
```

## 动手实现（Build It）

`code/main.py` 用纯 Python 实现差分注意力。带已知信号加噪声结构的玩具查询让你直接测量噪声抵消比。

### 步骤 1：标准 softmax 注意力（Step 1: standard softmax attention）

Stdlib 矩阵运算：嵌套列表、手写矩阵乘、带减最大值数值稳定技巧的 softmax。

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### 步骤 2：把 Q、K 拆成两半（Step 2: split Q, K into two halves）

V1 风格：把头维减半。V2 风格：保持头维并把头数加倍。玩具实现用 V1 是为了教学清晰——数学相同，只是记账不同。

### 步骤 3：两条 softmax 分支 + 减法（Step 3: two softmax branches + subtraction）

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

注意：输出权重可以为负。这没问题——值缓存仍然处理带符号贡献。后续 V 投影会吸收符号。

### 步骤 4：噪声抵消测量（Step 4: noise cancellation measurement）

构造长度为 1024 的合成序列。把信号 token 放在已知位置，其余填噪声。计算 (a) 标准 softmax 注意力在信号位置上的权重，以及 (b) 差分注意力权重。测量各自的信噪比。差分注意力稳定地给出更高的信噪比，倍数约 3×–10×，取决于两条分支被训练得有多不同。

### 步骤 5：V1 vs V2 参数记账（Step 5: V1 vs V2 parameter accounting）

给定配置（hidden=4096，heads=32，d_head=128），打印：

- 基线 Transformer：Q、K、V 各为 `hidden * hidden`，MLP 为 4 * hidden。
- DIFF V1：Q、K 各为 `hidden * hidden`，V 为 `hidden * hidden`（不变），内部头维减半。增加按头 `lambda` 参数（O(heads * d_head)）。
- DIFF V2：Q 为 `2 * hidden * hidden`，K 为 `hidden * hidden`，V 为 `hidden * hidden`。额外维度在 O_W 之前投影回去。增加同样的 `lambda` 参数。

玩具实现测量 V2 的额外参数成本（每个注意力块大约多 `hidden * hidden`）并打印出来。

## 实际应用（Use It）

截至 2026 年 4 月，DIFF V2 尚未装进每一台生产推理服务器，但 vLLM 和 SGLang 的集成正在进行。与此同时，该模式出现在：

- Microsoft 内部的长上下文生产模型。
- 若干瞄准 256k 以上上下文的开放模型训练复现。
- 把 DIFF 注意力与交替层上的滑动窗口注意力结合起来的混合架构。

2026 年你会在何时伸手去拿它：

- 从零训练一个瞄准 64k 以上有效上下文的新模型。从一开始就加入差分注意力；事后再训代价昂贵。
- 微调一个长上下文模型，而“迷失在中间”失败主导你的评估。对 Q 投影做 LoRA 可以近似 DIFF 结构。

何时不该用：

- 你在服务一个长上下文表现已经稳定的预训练稠密模型。在已有权重上重训，成本很少能回本。
- 你的上下文始终低于 16k。噪声底可以忽略。

## 交付产物（Ship It）

本课产出 `outputs/skill-diff-attention-integrator.md`。给定模型架构、目标上下文长度、幻觉画像和训练预算，它产出一份把差分注意力加入新预训练或 LoRA 微调的集成计划。

## 练习（Exercises）

1. 运行 `code/main.py`。验证合成查询上差分注意力报告的信噪比高于标准 softmax 注意力。改变噪声幅度，并找出标准注意力变得不可用的交叉点。

2. 对 7B 级模型（hidden=4096，heads=32，d_head=128，32 层）计算从基线到 DIFF V1、以及从基线到 DIFF V2 的参数量增量。指出哪些组件增加了参数、哪些保持不变。

3. 阅读 DIFF V1 论文第 3 节（arXiv:2410.05258）以及 DIFF V2 Hugging Face 博客第 2 节。用两句话解释：为什么 V1 的按头 RMSNorm 是必要的，以及为什么 V2 可以去掉它而不引起训练发散。

4. 实现一次消融：用 `lambda = 0`（纯第一条 softmax）和 `lambda = 1`（完全减法）计算差分注意力。在合成查询上测量信噪比如何随扫描变化。找出使信噪比最大的 `lambda`。

5. 把玩具扩展到 GQA + DIFF V2。选 8 个 KV 头和 32 个 Q 头。证明 KV 缓存大小与同样 (8, 32) 配置的基线 GQA 模型匹配。

## 关键术语（Key Terms）

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Differential attention | “两个 softmax 相减” | 把 Q、K 拆成两半，算两张 softmax 图，用 lambda 缩放后从第一张减去第二张，再乘 V |
| Noise floor | “softmax 非零的尾巴” | softmax 给每个无关 token 的 O(1/N) 权重，在长上下文上加总为 O(1) |
| lambda | “减法的尺度” | 按头可学习标量，参数化为 `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`；可以为负 |
| DIFF V1 | “ICLR 2025 那个版本” | 原始差分 Transformer；为保持参数量把头维减半，需要自定义内核，解码更慢 |
| DIFF V2 | “2026 年 1 月的修复” | 加倍 Q 头、保持 KV 头；匹配基线解码速度，并能与 FlashAttention 一起工作 |
| Per-head RMSNorm | “V1 的稳定器” | V1 在差分之后施加的额外归一化；V2 去掉它以防后期训练失稳 |
| Signal-to-noise ratio | “有多少注意力被浪费” | 真正信号位置上的权重与无关位置平均权重之比 |
| Lost in the middle | “长上下文失败模式” | 检索精度在长上下文中间文档处下陷的经验现象——差分注意力会减轻它 |
| Arithmetic intensity | “每加载一字节的 FLOPs” | V2 通过每次 KV 加载加倍查询而在解码时提高的比率；对内存受限解码很重要 |

## 延伸阅读（Further Reading）

- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — 带噪声抵消理论与长上下文消融的原始论文
- [Microsoft unilm — Differential Transformer V2 (Hugging Face blog, January 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) — 生产栈重写，匹配基线解码，兼容 FlashAttention
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) — 为何减法能恢复预训练注意力结构的理论分析
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) — 参数共享变体
- [Vaswani et al. — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) — DIFF 从中做减法的基线 Transformer
- [Liu et al. — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) — 差分注意力瞄准的长上下文基准
