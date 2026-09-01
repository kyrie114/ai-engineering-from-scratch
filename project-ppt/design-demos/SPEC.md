# 设计 Spec · Phase 01 阶段导读 deck 两页代表页(三方向共同输入)

## 产品是什么
「AI Engineering from Scratch」(github.com/rohitg00/ai-engineering-from-scratch,MIT,511 课/20 阶段/~329h)
的**阶段导读 deck**。本 spec 只做其中 **Phase 01 — Math Foundations(数学地基)** 的两页代表页;
选定方向后将按同一设计语言铺满全部 20 个阶段。

## 受众与场景
中文技术学习者与分享场合;投屏观看(观众距离 1-10m);deck 要横向翻页,每页固定 1920×1080。

## 核心信息(两页的真实内容,禁止虚构/占位文案)

**Page 1 · 封面**
- 眉头:PHASE 01 / 20 · AI ENGINEERING FROM SCRATCH
- 主标:Math Foundations(EN)· 数学地基(CN)
- 一句话:每个算法背后,都是一段数学。直觉先于公式,用代码把直觉造出来。
- 元数据:22 LESSONS · ~32 HOURS · PYTHON + JULIA · README 原话 "The intuition behind every AI algorithm, through code."
- 出品:MIT · github.com/rohitg00/ai-engineering-from-scratch · 2026.08

**Page 2 · 课单与体量(数据页)**
- 标题区:这一阶段的 22 课(课单 1/2,展示 L01-L12)
- 真实课单(Lxx · EN title · Type · Lang):
  L01 Linear Algebra Intuition · Learn · Python, Julia
  L02 Vectors, Matrices & Operations · Build · Python, Julia
  L03 Matrix Transformations & Eigenvalues · Build · Python, Julia
  L04 Calculus for ML: Derivatives & Gradients · Learn · Python
  L05 Chain Rule & Automatic Differentiation · Build · Python
  L06 Probability & Distributions · Learn · Python
  L07 Bayes' Theorem & Statistical Thinking · Build · Python
  L08 Optimization: Gradient Descent Family · Build · Python
  L09 Information Theory: Entropy, KL Divergence · Learn · Python
  L10 Dimensionality Reduction: PCA, t-SNE, UMAP · Build · Python
  L11 Singular Value Decomposition · Build · Python, Julia
  L12 Tensor Operations · Build · Python
- 数据:22 lessons · ~32h(docs 标注合计)· quiz 110 题 · outputs 工件 23 件
- 方法论脊柱(需在页面出现):MOTTO → PROBLEM → CONCEPT → BUILD IT(裸写)→ USE IT(NumPy/PyTorch 对照)→ SHIP IT(带走工件)
- 尾注:+ 10 MORE · 完整清单见 README 与课程站

## 情感基调
从零手写的工程严谨感;可信赖的教科书气质;克制但要有作者性,拒绝「通用 AI 生成的演示」。

## 输出格式与尺寸(硬约束)
- 每方向产出 **2 个独立单文件 HTML**(cover 与 data 各一),每文件内一个固定 1920×1080 的 section/body,overflow:hidden,零滚动。
- 纯 HTML/CSS(内联 SVG 允许),JS 可省;Google Fonts 走 CDN link 可用。
- **单页内容不自带页码/进度条/导航栏**(这些由后续 deck 外壳统一承载)。

## 已知约束
- 中文为主 + 英文技术术语;文案用「」引号;不得出现 Lorem ipsum /「标题文字」占位。
- 可读性硬底线(投屏):正文 ≥14px、标签/注释 ≥12px、文字对比度 ≥4.5:1。
- 禁区:不做 GitHub-dark(#0D1117+霓虹 glow);不堆装饰渐变;不用 emoji 当图标;不做圆角卡片+左彩条 accent 的默认套路(除非该风格 DNA 本身如此且注明理由)。

## 图片需求(Phase 3.5 判定)
数据/结构型内容,**无需照片**。真实品牌资产只有一个:仓库 banner(`D:\pycharm_code\AI\ai-engineering-from-scratch\assets\banner.svg`,
电光蓝 #3553ff + 纸纹 + 近黑,自述 "reference manual banner")。各方向按自己的语言决定用法:
可直接内联该 SVG、或只采样其颜色、或说明为何不用(三选一,须写明)。

## 色彩协议(三步法,必做)
采样(品牌资产/内容真图/文化语境)→ 收敛(oklch 思路压到 2-3 个有彩色+中性明度序列)→
**写一句「为什么是这个色」的论证放进交付说明**。直接抄现成 hex 而写不出论证 = 不合格。

## 视觉母题(内容独有的 form 种子)
「六拍结构 MOTTO→PROBLEM→CONCEPT→BUILD IT→USE IT→SHIP IT」的循环/进程感;
「22 课的序列与刻度」;「从裸数学到生产框架的双轨对照」。三方向必须从这个母题长出各自的构图。
