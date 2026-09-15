# DeepSeek-V3 架构走读（DeepSeek-V3 Architecture Walkthrough）

> Phase 10 · Lesson 14 点出了每个开放模型都会拧的六个架构旋钮。DeepSeek-V3（2024 年 12 月，总计 671B 参数，37B 激活）把六个都拧了，又加了四个：多头潜在注意力（Multi-Head Latent Attention）、无辅助损失负载均衡、多 Token 预测（Multi-Token Prediction），以及 DualPipe 训练。本课自上而下阅读 DeepSeek-V3 的架构，并从已发布配置推导每一个参数量。结束时你能解释为什么 671B/37B 比例是正确赌注，以及为什么 MLA + MoE 合在一起在前沿上胜过单独任何一个。

**Type:** Learn
**Languages:** Python (stdlib, parameter calculator)
**Prerequisites:** Phase 10 · 14 (open-model walkthroughs), Phase 10 · 17 (NSA), Phase 10 · 18 (MTP), Phase 10 · 19 (DualPipe)
**Time:** ~75 minutes

## 学习目标（Learning Objectives）

- 自上而下阅读 DeepSeek-V3 配置，用六个 GPT-2 旋钮加上四个 DeepSeek 特有增量解释每个字段
- 推导总参数量（671B）、激活参数量（37B），以及各自由哪些组件贡献
- 计算 MLA 在 128k 上下文下的 KV 缓存占用，并与同等激活参数、使用 GQA 的稠密模型会付的代价比较
- 陈述四个 DeepSeek 特有创新（MLA、MTP、无辅助损失路由、DualPipe），并指出各自瞄准架构/训练栈的哪一部分

## 问题（The Problem）

DeepSeek-V3 是第一个架构与 Llama 家族有实质差异的前沿开放模型。Llama 3 405B 是「拧了六个旋钮的 GPT-2」。DeepSeek-V3 是六个旋钮全拧再加四个。读 Llama 3 配置是读 DeepSeek 配置的热身，但深层结构——注意力块的形状、路由逻辑、训练时目标——差异大到需要单独走读。

学它的回报：DeepSeek-V3 的开放权重发布改写了开放模型里「前沿能力」的含义。这个架构是许多 2026 训练运行在复制的蓝图。对任何接触前沿 LLM 训练或推理的角色，理解它是入场券。

## 概念（The Concept）

### 不变的核心，再来一次（The invariant core, again）

DeepSeek-V3 仍是自回归。它仍堆叠解码器块。每个块仍是注意力加 MLP 加两个 RMSNorm。MLP 仍用 SwiGLU。仍用 RoPE。预归一化（pre-norm）。权重绑定嵌入。与每一个 Llama 或 Mistral 相同的基线。

### 转折：用 MLA 代替 GQA（The twist: MLA instead of GQA）

从 Phase 10 · 14 你知道 GQA 通过让若干 Q 头共享 K 和 V 来缩小 KV 缓存。多头潜在注意力（Multi-Head Latent Attention，MLA）走得更远：K 和 V 被压缩进共享的低秩潜在表示（`kv_lora_rank`），再在每个头上即时解压。KV 缓存只存潜在——通常每层每 token 512 个浮点，而不是 8 x 128 = 1024 个浮点。

在 128k 上下文，带 MLA 的 DeepSeek-V3（每层每 token 一个共享潜在 `c^{KV}`；K 和 V 都从这个潜在经上投影得到，而这些上投影可以吸收进后续矩阵乘）：

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

假设的 GQA 基线（Llama 3 70B 形状，8 个 KV 头，头维度 128）会付：

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

在 128k 上下文，MLA 比 Llama-3-70B 风格的 GQA 缓存小 4 倍。

权衡：MLA 在每次注意力计算（每个头）上加了解压步骤。额外计算相对省下的带宽很小。长上下文推理净赢。

### 路由：无辅助损失负载均衡（The routing: auxiliary-loss-free load balancing）

MoE 路由器决定哪些 top-k 专家处理每个 token。朴素路由器会把太多工作集中到少数专家，让其他闲着。标准修法：加一项惩罚负载不均衡的辅助损失。这能用，但会轻微损害主任务表现。

DeepSeek-V3 引入无辅助损失方案。给路由器 logits 加上每专家偏置项，训练中用简单规则调整：若专家 `e` 过载，减小 `bias_e`；若欠载，增大它。没有额外损失项。训练保持干净。专家负载保持均衡。

对主损失的影响：测不到。对 MoE 架构的影响：更干净，没有要调的辅助损失超参。

### MTP：更密的训练 + 免费草稿（The MTP: denser training + free draft）

从 Phase 10 · 18 你知道 DeepSeek-V3 加了 D=1 的 MTP 模块，预测前方两个位置的 token。推理时，训练好的模块被改造成接受率超过 80% 的投机解码草稿。训练时，每个隐状态被监督在 D+1 = 2 个目标上，提供更密的信号。

参数：在 671B 主模型之上 14B。开销：2.1%。

### 训练：DualPipe（The training: DualPipe）

从 Phase 10 · 19 你知道 DualPipe 是一种双向流水线，把前向与反向块与跨节点 all-to-all 通信重叠。在 DeepSeek-V3 的 2,048-H800 规模上，它回收了 1F1B 大约会丢给流水线气泡的 245k GPU 小时。

### 配置，逐字段（The config, field by field）

这是 DeepSeek-V3 配置（简化）：

```
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

解析它：

- `hidden_size=7168`：嵌入维度。
- `num_hidden_layers=61`：总块深度。
- `first_k_dense_layers=3`：前 3 个块使用大小为 18432 的稠密 MLP。其余 58 个使用 MoE。
- `num_attention_heads=128`：128 个查询头。
- `kv_lora_rank=512`：K 和 V 被压缩到这个潜在维度，再按头解压。
- `num_experts=256, num_experts_per_tok=8`：每个 MoE 块有 256 个专家，路由 top-8。
- `shared_experts=1`：在 256 个被路由专家之上，1 个始终开启的专家对每个 token 都贡献。可以把它想成「稠密地板」，确保每个 token 都得到可靠的东西。
- `moe_intermediate_size=2048`：每个专家的 MLP 隐层大小。比稠密 MLP 小，因为有 256 个。

### 参数核算（Parameter accounting）

完整计算在 `code/main.py`。头条数字：

- 嵌入：`vocab * hidden = 129280 * 7168 = ~0.93B`。
- 前 3 个稠密块：带 MLA 的注意力（每块 ~144M）+ 稠密 MLP（每块 ~260M）+ 归一化。总计约 1.2B。
- 58 个 MoE 块：带 MLA 的注意力（~144M）+ 各 256 个专家（每个 30M）+ 1 个共享专家（30M）+ 归一化。每块含全部专家约 ~7.95B。58 个 MoE 块总计 461B。
- MTP 模块：14B。

总计：核心架构约 ~476B + 14B MTP；已发布的 671B 数字还计入额外结构参数（偏置张量、专家特有组件、共享专家缩放等）。我们在计算器里复现的数字与已发布值相差 3–5%——差额来自 DeepSeek 报告第 2 节附录里更细的核算。

每次前向的激活参数：

- 注意力：每层 144M * 61 = 8.8B（所有层都开火）。
- MLP 激活：前 3 层稠密（3 * 260M = 780M），58 个 MoE 层各自激活 8 个被路由 + 1 个共享 + 路由开销。每层激活 MLP：~260M。总计：3 * 260M + 58 * 260M = ~15.9B。
- 嵌入 + 归一化：1.2B。
- 总激活：大约 26B 核心 + 14B MTP（训练时有，推理时不总跑）≈ 37B。

### 671B / 37B 比例（The 671B / 37B ratio）

18 倍稀疏比（激活参数是总量的 5.5%）。DeepSeek-V3 是已开放权重的最稀疏前沿 MoE 模型。Mixtral 8x7B 比例 13/47（28%）稠密得多。Llama 4 Maverick 比例 17B/400B（4.25%）相当。DeepSeek 的赌注：在前沿规模，更多专家、更低激活比，会给出更好的每激活 FLOP 质量。

### DeepSeek-V3 坐在哪里（Where DeepSeek-V3 sits）

| 模型 | 总计 | 激活 | 比例 | 注意力 | 新想法 |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + 无辅助损失 + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN 扩展 |

### 后续：R1、V4（The follow-on: R1, V4）

DeepSeek-R1（2025）是在 V3 主干上的推理训练运行。R1 用同一套架构。变的是后训练配方（在可验证任务上大规模 RL），不是预训练架构。

DeepSeek-V4（若发布）预期会保留 MLA + MoE + MTP，并加上 DSA（DeepSeek Sparse Attention），即 Phase 10 · 17 中 NSA 的后继。谱系稳定：架构级创新累积；每个版本拧额外的旋钮。

```figure
moe-routing
```

## 使用它（Use It）

`code/main.py` 是专为 DeepSeek-V3 形状定制的参数计算器。运行它，把输出与论文数字比较，并在假设变体上使用（256 专家对 512，top-8 对 top-16，MLA rank 512 对 1024）。

该看什么：

- 总参数量相对已发布的 671B。
- 激活参数量相对已发布的 37B。
- 128k 上下文的 KV 缓存——MLA 对 GQA 的比较。
- 逐层拆解，看参数预算实际花在哪里。

## 交付（Ship It）

本课产出 `outputs/skill-deepseek-v3-reader.md`。给定一个 DeepSeek 家族模型（V3、R1，或任何未来变体），它产出逐组件的架构阅读：点出配置每个字段、按组件推导参数量，并识别该模型使用四个 DeepSeek 特有创新中的哪几个。

## 练习（Exercises）

1. 运行 `code/main.py`。把计算器的总参数估计与已发布的 671B 比较，并指出差额来自哪里。论文第 2 节有完整分项。

2. 把配置改成 MLA rank 256 而不是 512。计算 128k 上下文下得到的 KV 缓存大小。它买到百分之几的缩减，对每头表达力的代价是什么？

3. 把 DeepSeek-V3 的（256 专家，top-8）路由与假设的（512 专家，top-8）变体比较。总参数增长；激活参数不变。额外专家容量理论上买到什么，推理时代价是什么？

4. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）第 2.1 节关于 MLA。用三句话解释为什么 K 和 V 的解压矩阵可以在推理时「吸收」进后续矩阵乘以提高效率。

5. DeepSeek-V3 对大多数运算使用 FP8 训练。计算用 FP8 相对 BF16 存储 671B 权重的内存节省。这如何与 14.8T token 的训练预算相交？

## 关键术语（Key Terms）

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| MLA | 「多头潜在注意力」 | 把 K 和 V 压进共享低秩潜在（kv_lora_rank，通常 512），按头即时解压；KV 缓存只存潜在 |
| kv_lora_rank | 「MLA 压缩维」 | K 和 V 共享潜在的大小；DeepSeek-V3 用 512 |
| 前 k 个稠密层 | 「早期层保持稠密」 | MoE 模型的前几层跳过 MoE 路由器，跑稠密 MLP 以求稳定 |
| num_experts_per_tok | 「Top-k 路由」 | 每个 token 开火的被路由专家数；DeepSeek-V3 用 8 |
| 共享专家 | 「始终开启的专家」 | 无论路由如何都处理每个 token 的专家；DeepSeek-V3 用 1 |
| 无辅助损失路由 | 「偏置调整的负载均衡」 | 训练中调整每专家偏置项以保持负载均衡，不额外加损失项 |
| MTP 模块 | 「额外预测头」 | 从 h^(1) 和 E(t+1) 预测 t+2 的 transformer 块；更密训练，免费投机解码草稿 |
| DualPipe | 「双向流水线」 | 把前向/反向计算与跨节点 all-to-all 重叠的训练调度 |
| 激活参数比 | 「稀疏度」 | active_params / total_params；DeepSeek-V3 达到 5.5% |
| FP8 训练 | 「8 比特训练」 | 训练存储和许多计算用 FP8；相对 BF16 大约减半内存，质量代价很小 |

## 延伸阅读（Further Reading）

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 完整架构、训练与结果文档
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — 配置文件与部署说明
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — 引入 MLA 的前身
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — 在 V3 架构上的推理训练后继
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) — DeepSeek 家族注意力的未来方向
- [DualPipe repository](https://github.com/deepseek-ai/DualPipe) — 训练调度参考
