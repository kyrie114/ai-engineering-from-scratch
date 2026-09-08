# LLM 评估——RAGAS, DeepEval, G-Eval（LLM Evaluation — RAGAS, DeepEval, G-Eval）

> 精确匹配和 F1 错过了语义等价。人工审查无法扩展。LLM 作为评判是生产答案——有足够的校准来信任这个数字。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time:** ~75 minutes

## 问题（The Problem）

你的 RAG 系统回答："June 29th, 2007。"
黄金参考是："June 29, 2007。"
精确匹配得 0 分。F1 约 75%。人类会打 100 分。

现在乘以 10,000 个测试用例。再乘以对检索器、分块、提示或模型的每一次更改。你需要一个评估器理解含义，廉价地大规模运行，不撒谎关于回归，并 surfaced 正确的失败模式。

2026 年有三个框架拥有这个问题。

- **RAGAS。** 检索增强生成评估（Retrieval-Augmented Generation ASsessment）。四个 RAG 度量（忠实度、答案相关性、上下文精确率、上下文召回率），具有 NLI + LLM 评判后端。研究支持，轻量级。
- **DeepEval。** LLM 的 pytest。G-Eval、任务完成、幻觉、偏差度量。CI/CD 原生。
- **G-Eval。** 一种方法（也是一个 DeepEval 度量）：LLM 作为评判，具有思维链、自定义标准、0-1 分数。

三者都依赖 LLM 作为评判。本课为这种方法建立直觉以及围绕它的信任层。

## 概念（The Concept）

![Four evaluation dimensions, LLM-as-judge architecture](../assets/llm-evaluation.svg)

**LLM 作为评判。** 用 LLM 替换静态度量，给定评分标准对输出评分。给定 `(query, context, answer)`，提示评判 LLM："Score 0-1 on faithfulness。"返回分数。

为什么它有效：LLM 以人类判断的一小部分成本近似人类判断。GPT-4o-mini 每个评分案例约 $0.003，使 1000 样本的回归评估运行低于 $5。

为什么它静默失败：

1. **评判偏见。** 评判喜欢更长的答案、来自自己模型家族的答案、匹配提示风格的答案。
2. **JSON 解析失败。** 错误的 JSON → NaN 分数 → 从聚合中静默排除。RAGAS 用户知道这种痛苦。用 try/except + 显式失败模式来门控。
3. **模型版本的漂移。** 升级评判会改变每个度量。冻结评判模型 + 版本。

**RAG 四要素。**

| 度量 | 问题 | 后端 |
|--------|----------|---------|
| 忠实度 | 答案中的每个主张是否来自检索的上下文？ | 基于 NLI 的蕴涵 |
| 答案相关性 | 答案是否解决了问题？ | 从答案生成假设问题；与真实问题比较 |
| 上下文精确率 | 在检索到的块中，有多少实际上相关？ | LLM 评判 |
| 上下文召回率 | 检索是否返回了所有需要的内容？ | 对黄金答案的 LLM 评判 |

**G-Eval。** 定义一个自定义标准："答案是否引用了正确的来源？"框架自动扩展为思维链评估步骤，然后评分 0-1。适合 RAGAS 未覆盖的领域特定质量维度。

**校准。** 在你有人工标签的 100 个样本上运行之前，不要信任原始评判分数。绘制评判 vs 人工。计算 Spearman rho。如果 rho < 0.7，你的评判评分标准需要工作。

```figure
n5-judge-gauge
```

## 构建它（Build It）

### 步骤 1：使用 NLI 的忠实度（RAGAS 风格）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

将答案分解为原子主张。根据检索的上下文对每个主张进行 NLI 检查。忠实度 = 被支持的比例。

### 步骤 2：答案相关性

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

如果答案暗示的问题与被问的不同，相关性就会下降。

### 步骤 3：G-Eval 自定义度量

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

评估步骤是评分标准。显式步骤比隐式的 "score 0-1" 提示更稳定。

### 步骤 4：CI 门控

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

作为 pytest 文件交付。在每次 PR 上运行。在回归上阻止合并。

### 步骤 5：从头开始的玩具评估

见 `code/main.py`。仅标准库的忠实度（答案主张与上下文重叠）和相关性（答案词元与问题词元重叠）近似。不用于生产。展示形状。

## 陷阱

- **没有校准。** 与人工标签相关性 0.3 的评判是噪音。在交付之前要求校准运行。
- **自我评估。** 使用相同的 LLM 生成和评判会将分数提高 10-20%。为评判使用不同的模型家族。
- **成对评判中的位置偏见。** 评判更喜欢呈现的第一个选项。始终随机化顺序并双向运行。
- **原始聚合隐藏失败。** 平均分数 0.85 经常隐藏 5% 的灾难性失败。始终检查下分位数。
- **黄金数据集腐烂。** 随时间的漂移破坏纵向比较的未版本化评估集。用每次更改标记数据集。
- **LLM 成本。** 在大规模时，评判调用主导成本。使用满足校准阈值的最便宜的模型。GPT-4o-mini、Claude Haiku、Mistral-small。

## 使用它（Use It）

2026 年技术栈：

| 使用场景 | 框架 |
|---------|-----------|
| RAG 质量监控 | RAGAS（4 个度量） |
| CI/CD 回归门控 | DeepEval + pytest |
| 自定义领域标准 | DeepEval 中的 G-Eval |
| 在线实时流量监控 | 免参考模式的 RAGAS |
| 人在回路抽查 | 带注释 UI 的 LangSmith 或 Phoenix |
| 红队 / 安全评估 | Promptfoo + DeepEval |

典型栈：RAGAS 用于监控，DeepEval 用于 CI，G-Eval 用于新维度。运行全部三个；它们有用地不同意。

## 发布它（Ship It）

保存为 `outputs/skill-eval-architect.md`：

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## 练习（Exercises）

1. **简单（Easy）。** 在 10 个具有已知幻觉的 RAG 示例上使用 RAGAS。验证忠实度度量捕捉到每一个。
2. **中等（Medium）。** 手工标记 50 个 QA 答案 0-1 正确性。使用 G-Eval 评分。测量评判与人工的 Spearman rho。
3. **困难（Hard）。** 构建一个带有 DeepEval 的 pytest CI 门控。故意使检索器退化。验证门控失败。通过阈值检查最低 10% 添加下分位数警报。

## 关键术语（Key Terms）

| 术语（Term） | 人们怎么说（What people say） | 它实际意味着什么（What it actually means） |
|------|-----------------|-----------------------|
| LLM 作为评判 | 用 LLM 评分 | 给定评分标准提示评判模型对输出评分 0-1。 |
| RAGAS | RAG 度量库 | 具有 4 个免参考 RAG 度量的开放评估框架。 |
| 忠实度 | 答案有依据吗？ | 被检索上下文支持的答案主张比例。 |
| 上下文精确率 | 检索的块相关吗？ | 实际重要的 top-K 块比例。 |
| 上下文召回率 | 检索找到了一切吗？ | 被检索块支持的黄金答案主张比例。 |
| G-Eval | 自定义 LLM 评判 | 评分标准 + 思维链评估步骤 + 0-1 分数。 |
| 校准 | 信任但验证 | 评判分数与人工分数之间的 Spearman 相关性。 |

## 延伸阅读（Further Reading）

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — RAGAS 论文。
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — G-Eval 论文。
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) — 开放生产栈。
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — 偏见、校准、限制。
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — 集成 RAGAS、DeepEval、Phoenix 的统一框架。
