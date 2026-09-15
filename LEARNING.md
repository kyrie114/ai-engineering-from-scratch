# My AI Engineering Path
<!-- Managed by the ai-engineering-from-scratch learning skills.
     Repo: https://github.com/rohitg00/ai-engineering-from-scratch -->

## Mission
职业转型:以 AI 工程师为目标,系统补齐底层功底。终极目标:从零训练一个模型——手写 backprop 和 attention,亲眼看一个小模型训出来。

## Placement
- Date: 2026-09-01
- Score: self-selected(跳过摸底测验,按实际进度定位:Phase 1 的 L01–L09 已完成,L10 进行中)
- Entry point: Phase 1: Math Foundations(从 L10 Dimensionality Reduction 继续)
- Pace: 越快越好(冲刺模式,能挤的时间都投)

## Path
| Phase | Name | Status | Est. hours |
|-------|------|--------|------------|
| 0 | Setup & Tooling | Skip | ~14h |
| 1 | Math Foundations | Do | ~23h |
| 2 | ML Fundamentals | Do | ~21h |
| 3 | Deep Learning Core | Do | ~15h |
| 4 | Computer Vision | Do | ~27h |
| 5 | NLP — Foundations to Advanced | Do | ~30h |
| 6 | Speech & Audio | Do | ~18h |
| 7 | Transformers Deep Dive | Do | ~14h |
| 8 | Generative AI | Do | ~14h |
| 9 | Reinforcement Learning | Do | ~13h |
| 10 | LLMs from Scratch | Do | ~26h |
| 11 | LLM Engineering | Do | ~17h |
| 12 | Multimodal AI | Do | ~65h |
| 13 | Tools & Protocols | Do | ~43h |
| 14 | Agent Engineering | Do | ~42h |
| 15 | Autonomous Systems | Do | ~20h |
| 16 | Multi-Agent & Swarms | Do | ~28h |
| 17 | Infrastructure & Production | Do | ~32h |
| 18 | Ethics, Safety & Alignment | Do | ~31h |
| 19 | Capstone Projects | Do | ~620h |

Phase 0 跳过理由:学习者的 Python/numpy 环境已在正常运行,课程代码已实际跑通。

## Progress log
| Date | Lesson | Quiz | Note |
|------|--------|------|------|
| ≤2026-08-29 | phase01/01–08 | — | 建档前已完成,补记;当时分数未留档 |
| 2026-08-30 | phase01/09-information-theory | 3/3 | KL≈CE 通过;PPL、MI 当场重讲后过关;MI-vs-Pearson 仍是重复薄弱点 |
| 2026-09-05 | phase01/11-singular-value-decomposition | 3/3 | 完整覆盖:概念(rotate-scale-rotate) + power iteration 从零实现 + 官方 demo 验证 + 3 道 post 全对 |
| 2026-09-02 | phase01/10-dimensionality-reduction | 3/3 | 收官:概念+实验(有损/无损记账、kernel PCA 同心圆、elbow 侦探)+ 3 道 post 全对 |
| 2026-09-10 | phase01/12-tensor-operations | 3/3 | broadcast 初期卡住("一点都不会"),讲解后通过;transpose/reshape 内存顺序理解正确 |
| 2026-09-14 | phase01/13-numerical-stability | 2/3 | 稳定 softmax 的减 max 误当成零均值; catastrophic cancellation / bfloat16 清楚 |
| 2026-09-14 | phase01/14-norms-and-distances | 1/3 | 课中余弦/LASSO 直觉对; post 误选 Wasserstein「总更小」、马氏当成比字符串 |
| 2026-09-14 | phase01/14-norms-and-distances (重修) | — | 慢讲后收束全对: 余弦vsL2、马氏=协方差、W vs KL=不重叠仍有梯度; 承认上次刷题式教学无效 |
| 2026-09-14 | phase01/15-statistics-for-ml | 2/3 | p值/Pearson·Spearman/bootstrap 先「不知道」后讲透; 多样本假阳性误当成仍是5%; 显著≠实用、bootstrap优势清楚 |

## Review queue
- (空; L14 马氏/Wasserstein 已在重修销账)
