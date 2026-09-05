# 采样方法（Sampling Methods）

> 采样是 AI 探索可能性空间的方式。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06-07 (Probability, Bayes' Theorem)
**Time:** ~120 minutes

## 学习目标（Learning Objectives）

- 仅使用均匀随机数，从零实现逆 CDF 采样、拒绝采样和重要性采样
- 为语言模型的词元生成构建温度采样、top-k 采样和 top-p（核）采样
- 解释重参数化技巧（reparameterization trick），以及为什么它让变分自编码器（VAE）中的采样过程能够反向传播
- 运行 Metropolis-Hastings MCMC，从非归一化的目标分布中采样

## 问题（The Problem）

语言模型处理完你的提示词之后，产出一个包含 50,000 个 logits 的向量——词表中的每个词元对应一个。现在它必须从中挑出一个。怎么做？

如果它总是挑概率最高的词元，那么每次回答都一模一样：确定、无趣。如果它完全随机地挑，输出就是乱码。答案介于这两个极端之间，而这个"之间"由采样控制。

采样并不局限于文本生成。强化学习通过采样轨迹来估计策略梯度。VAE 通过从学到的分布中采样并对随机性做反向传播来学习潜在表示。扩散模型通过采样噪声并迭代去噪来生成图像。蒙特卡洛方法估计没有闭式解的积分。MCMC 算法探索无法枚举的高维后验分布。

每一个生成式 AI 系统都是一个采样系统。采样策略决定了输出的质量、多样性和可控性。本课从零构建每一种主流采样方法，从均匀随机数开始，到驱动现代 LLM 和生成模型的技术为止。

## 核心概念（The Concept）

### 为什么采样重要（Why Sampling Matters）

采样在 AI 和机器学习中扮演四种基本角色：

**生成（Generation）。** 语言模型、扩散模型和 GAN 都通过采样产生输出。采样算法直接控制创造性、连贯性和多样性。温度、top-k 和核采样是工程师每天调节的旋钮。

**训练（Training）。** 随机梯度下降采样小批量（mini-batch）。Dropout 采样要失活的神经元。数据增强采样随机变换。重要性采样对样本重新加权，以降低强化学习（PPO、TRPO）中的梯度方差。

**估计（Estimation）。** 机器学习中的许多量没有闭式解：数据分布上的期望损失、能量模型的配分函数、贝叶斯推断中的证据。蒙特卡洛估计通过对样本求平均来近似所有这些量。

**探索（Exploration）。** MCMC 算法探索贝叶斯推断中的后验分布。进化策略采样参数扰动。汤普森采样（Thompson sampling）在多臂老虎机问题中平衡探索与利用。

核心挑战在于：你只能直接从简单分布（均匀分布、正态分布）中采样。其他一切情况下，你都需要一种方法，把简单样本转换成目标分布的样本。

### 均匀随机采样（Uniform Random Sampling）

每种采样方法都从这里出发。均匀随机数生成器产生 [0, 1) 区间内的值，其中任何等长的子区间概率相等。

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

要从 n 个离散项中均匀采样，生成 U 并返回 floor(n * U)。要从连续区间 [a, b] 中采样，计算 a + (b - a) * U。

关键洞察：一个均匀随机数恰好包含从任意分布产生一个样本所需的随机量。诀窍在于找到正确的变换。

### 逆 CDF 方法（逆变换采样）（Inverse CDF Method (Inverse Transform Sampling)）

累积分布函数（CDF）把值映射到概率：

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

逆 CDF 把概率映射回值。如果 U ~ Uniform(0, 1)，那么 X = F_inverse(U) 服从目标分布。

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**指数分布示例：**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

当你能写出 F_inverse 的闭式解时，这种方法完美适用。对于正态分布，不存在闭式的逆 CDF，所以我们要用其他方法（Box-Muller 或数值近似）。

**离散版本：** 对于离散分布，把 CDF 构建成累积和，生成 U，然后找到累积和首次超过 U 的那个下标。第 06 课的 `sample_categorical` 就是这么工作的。

### 拒绝采样（Rejection Sampling）

当你无法对 CDF 求逆、但能在相差一个常数的情况下计算目标 PDF（probability density function）时，拒绝采样就派上用场。

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

界 M 越紧，接受率越高。在低维（1 到 3 维）下拒绝采样效果很好。在高维下，接受率呈指数级下降，因为提议分布的大部分体积都会被拒绝。这就是拒绝采样的维度灾难（curse of dimensionality）。

**示例：从截断正态分布采样。** 在截断区间上使用均匀提议分布。包络 M 就是正态 PDF 在该区间上的最大值。

**示例：从半圆采样。** 在包围矩形内均匀提议。如果点落在半圆内就接受。蒙特卡洛法计算 pi 就是这么做的：接受率等于面积比 pi/4。

### 重要性采样（Importance Sampling）

有时你并不需要目标分布 p(x) 的样本，而是需要估计 p(x) 下的某个期望，而你手里只有另一个分布 q(x) 的样本。

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

这在强化学习中至关重要。在 PPO（Proximal Policy Optimization）中，你用旧策略 pi_old 采集轨迹，却想优化新策略 pi_new。重要性权重就是 pi_new(a|s) / pi_old(a|s)。PPO 会裁剪这些权重，防止新策略偏离旧策略太远。

重要性采样估计量的方差取决于 q 与 p 有多相似。如果 q 与 p 差别很大，少数样本会获得巨大的权重并主导估计。自归一化重要性采样通过除以权重总和来缓解这个问题：

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### 蒙特卡洛估计（Monte Carlo Estimation）

蒙特卡洛估计通过对随机样本求平均来近似积分。大数定律保证了收敛性。

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

误差率与维度无关。这就是为什么在基于网格的积分无法进行的高维场景中，蒙特卡洛方法占据主导地位。

**估算 pi：**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**估算期望：**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### 马尔可夫链蒙特卡洛（MCMC）：Metropolis-Hastings（Markov Chain Monte Carlo (MCMC): Metropolis-Hastings）

MCMC 构造一条平稳分布等于目标分布 p(x) 的马尔可夫链（Markov chain）。经过足够多步之后，这条链上的样本（近似）就是 p(x) 的样本。

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

对于对称提议（q(x'|x) = q(x|x')），接受比简化为 p(x')/p(x)。这就是最初的 Metropolis 算法。

**为什么有效。** 接受规则保证了细致平衡（detailed balance）：处于 x 并移动到 x' 的概率等于处于 x' 并移动到 x 的概率。细致平衡意味着 p(x) 是这条链的平稳分布。

**实践要点：**
- 预烧期（burn-in）：在链达到均衡之前丢弃早期样本
- 稀疏化（thinning）：每隔 k 个样本保留一个，以降低自相关
- 提议尺度：太小则链移动缓慢（接受率高、探索慢）；太大则大多数提议被拒绝（接受率低、原地打转）
- 高维下高斯提议的最优接受率约为 0.234

### 吉布斯采样（Gibbs Sampling）

吉布斯采样是 MCMC 在多元分布上的一种特例。它不一次在所有维度上提议移动，而是每次从条件分布中更新一个变量。

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

吉布斯采样要求你能从每个条件分布 p(x_i | x_{-i}) 中采样。对很多模型来说这很直接：
- 贝叶斯网络：条件分布由图结构直接给出
- 高斯混合：条件分布是高斯的
- Ising 模型：每个自旋的条件分布只依赖其邻居

接受率恒为 1（每个提议都被接受），因为从精确条件分布中采样自动满足细致平衡。

**局限。** 当变量高度相关时，吉布斯采样混合得很慢，因为一次只更新一个变量，无法沿对角方向大步穿越分布。

### 温度采样（用于 LLM）（Temperature Sampling (Used in LLMs)）

语言模型为词表中的每个词元输出 logits z_1, ..., z_V。Softmax 把它们转换为概率。温度（temperature）在 softmax 之前对 logits 进行缩放：

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**为什么有效。** 把 logits 除以小于 1 的 T 会放大 logits 之间的差异。如果 z_1 = 2、z_2 = 1，除以 T = 0.5 得到 z_1/T = 4 和 z_2/T = 2，差距变大。经过 softmax 之后，logit 最高的词元拿到大得多的份额。

**实践中：**
- T = 0.0：贪心解码，最适合事实性问答
- T = 0.3-0.7：略有创造性，适合代码生成
- T = 0.7-1.0：平衡，适合一般对话
- T = 1.0-1.5：创意写作、头脑风暴
- T > 1.5：越来越随机，几乎没用

温度不会改变哪些词元是可能的，它改变的是分配给每个词元的概率质量。

### Top-k 采样（Top-k Sampling）

Top-k 采样把候选集限制为概率最高的 k 个词元，然后重新归一化并从这个受限集合中采样。

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-k 防止模型选中词表分布长尾中那些极不可能的词元（错别字、无意义内容）。问题在于：k 是固定的，与上下文无关。当模型很自信（某个词元概率 95%）时，k = 40 仍然允许 39 个备选。当模型不确定（概率分散在 1000 个词元上）时，k = 40 又把合理的选项切掉了。

### Top-p（核）采样（Top-p (Nucleus) Sampling）

Top-p 采样动态调整候选集的大小。它不保留固定数量的词元，而是保留累积概率超过 p 的最小词元集合。

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

模型自信时，核采样（nucleus sampling）只保留少数词元（可能 2-3 个）。模型不确定时，它保留很多（可能 200 个）。这种自适应行为正是核采样通常比 top-k 产出更好文本的原因。

**常见组合：**
- 温度 0.7 + top-p 0.9：不错的通用设置
- 温度 0.0（贪心）：最适合确定性任务
- 温度 1.0 + top-k 50：Fan 等人（2018）原始论文的设置

Top-k 和 top-p 可以组合使用：先应用 top-k，再在剩下的集合上应用 top-p。

### 重参数化技巧（用于 VAE）（Reparameterization Trick (Used in VAEs)）

变分自编码器的学习方式是：把输入编码成潜在空间中的一个分布，从该分布中采样，再把样本解码回去。问题在于：你无法穿过一个采样操作做反向传播。

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

重参数化技巧把随机性与参数分离开：

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

这之所以有效，是因为 N(mu, sigma^2) 与 mu + sigma * N(0, 1) 同分布。关键洞察：把随机性搬到一个无参数的来源（epsilon）上，然后把样本表示成参数的可微变换。

**在 VAE 训练循环中：**
1. 编码器为每个输入输出 mu 和 log(sigma^2)
2. 采样 epsilon ~ N(0, 1)
3. 计算 z = mu + sigma * epsilon
4. 解码 z 以重建输入
5. 沿步骤 4、3、2、1 反向传播（之所以可行，是因为步骤 3 可微）

没有重参数化技巧，VAE 就无法用标准反向传播训练。正是这一个洞察让 VAE 变得实用。

### Gumbel-Softmax（可微分类别采样）（Gumbel-Softmax (Differentiable Categorical Sampling)）

重参数化技巧适用于连续分布（高斯分布）。对于离散的类别分布，我们需要另一种方法。Gumbel-Softmax 为类别采样提供了可微近似。

**Gumbel-Max 技巧（不可微）：**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax（可微近似）：**

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax 产生离散样本的连续松弛。输出是一个概率向量（软 one-hot），而不是硬 one-hot。梯度可以流过 softmax。训练时的前向传播中，你可以使用"直通"（straight-through）估计器：前向传播用硬 argmax，反向传播用软 Gumbel-Softmax 梯度。

**应用：**
- VAE 中的离散潜在变量
- 神经架构搜索（选择离散操作）
- 硬注意力机制
- 离散动作的强化学习

### 分层采样（Stratified Sampling）

标准蒙特卡洛采样可能碰巧在样本空间中留下空洞。分层采样把空间划分为层（stratum）并在每一层中采样，强制实现均匀覆盖。

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

与标准蒙特卡洛相比，分层采样的方差总是更低或相等：

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**应用：**
- 数值积分（拟蒙特卡洛）
- 训练数据划分（保证每一折中的类别均衡）
- 带分层的重要性采样（两种技术结合）
- NeRF（Neural Radiance Fields）沿相机光线使用分层采样

### 与扩散模型的联系（Connection to Diffusion Models）

扩散模型通过一个采样过程生成图像。前向过程在 T 步内向图像添加高斯噪声，直到它变成纯噪声。反向过程学习去噪，一步步恢复原始图像。

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

与本课方法的联系：
- 每个去噪步骤都使用重参数化技巧（采样噪声，应用确定性变换）
- 噪声调度 {alpha_t} 控制着一种温度退火
- 训练使用蒙特卡洛估计来近似 ELBO（证据下界）
- 扩散模型中的祖先采样（ancestral sampling）是一条马尔可夫链（每一步只依赖当前状态）

整个图像生成过程就是迭代采样：从噪声出发，每一步都在学到的去噪模型的条件下，采样一个噪声略小的版本。

```figure
monte-carlo-pi
```

## 动手构建（Build It）

### 步骤 1：均匀采样与逆 CDF 采样（Step 1: Uniform and inverse CDF sampling）

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

生成 10,000 个指数分布样本，验证均值等于 1/lambda。

### 步骤 2：拒绝采样（Step 2: Rejection sampling）

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

用拒绝采样从截断正态分布中抽取样本，并对样本画直方图来验证形状。

### 步骤 3：重要性采样（Step 3: Importance sampling）

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

用均匀提议分布估计正态分布下的 E[X^2]，与已知答案（mu^2 + sigma^2）比较。

### 步骤 4：蒙特卡洛法估算 pi（Step 4: Monte Carlo estimation of pi）

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### 步骤 5：Metropolis-Hastings MCMC（Step 5: Metropolis-Hastings MCMC）

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

从双峰分布（两个高斯的混合）中采样，并可视化这条链的轨迹。

### 步骤 6：吉布斯采样（Step 6: Gibbs sampling）

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### 步骤 7：温度采样（Step 7: Temperature sampling）

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

展示温度如何改变一组词元 logits 的输出分布。

### 步骤 8：Top-k 与 Top-p 采样（Step 8: Top-k and top-p sampling）

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### 步骤 9：重参数化技巧（Step 9: Reparameterization trick）

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

演示梯度可以流过重参数化采样，却无法流过直接采样。

### 步骤 10：Gumbel-Softmax（Step 10: Gumbel-Softmax）

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

展示降低温度如何使输出逼近 one-hot 向量。

完整实现和全部可视化代码在 `code/sampling.py` 中。

## 生产实践（Use It）

配合 NumPy 和 SciPy 的生产级版本：

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

要做大规模 MCMC，使用专用库：
- PyMC：带 NUTS（自适应 HMC）的完整贝叶斯建模
- emcee：集成 MCMC 采样器
- NumPyro/JAX：GPU 加速的 MCMC

这些都是你亲手实现过的。现在你知道那些库调用背后在做什么了。

## 练习（Exercises）

1. 为柯西分布实现逆 CDF 采样。CDF 为 F(x) = 0.5 + arctan(x)/pi。生成 10,000 个样本，画出直方图并与真实 PDF 对比。注意它的重尾（远离中心的极端值）。

2. 使用 Uniform(0, 1) 提议分布，通过拒绝采样从 Beta(2, 5) 分布生成样本。把接受的样本与真实 Beta PDF 画在一起。理论接受率是多少？

3. 分别用 1,000、10,000 和 100,000 个样本，用蒙特卡洛估计 sin(x) 在 0 到 pi 上的积分。比较每个量级下的误差。验证误差按 O(1/sqrt(N)) 缩放。

4. 实现 Metropolis-Hastings，从二维分布 p(x, y) ∝ exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2) 中采样。画出样本和链轨迹。尝试不同的提议标准差。

5. 构建一个完整的文本生成演示：给定一个 10 个词的词表及其 logits，分别用 (a) 贪心、(b) temperature=0.7、(c) top-k=3、(d) top-p=0.9 生成 20 个词元的序列。比较 5 次运行之间的输出多样性。

## 关键术语（Key Terms）

| 术语 | 人们怎么说 | 它的实际含义 |
|------|----------------|----------------------|
| 采样 | "抽取随机值" | 按概率分布生成值。所有生成式 AI 背后的机制 |
| 均匀分布 | "全都等可能" | [a, b] 中的每个值都有相同的概率密度 1/(b-a)。所有采样方法的起点 |
| 逆 CDF | "概率变换" | F_inverse(U) 把均匀样本转换成任何已知 CDF 分布的样本。精确且高效 |
| 拒绝采样 | "提议后接受/拒绝" | 从简单的提议分布生成，按与目标/提议比值成比例的概率接受。精确但浪费样本 |
| 重要性采样 | "对样本重新加权" | 用 q(x) 的样本、按 p(x)/q(x) 对每个样本加权，来估计 p(x) 下的期望。强化学习中 PPO 的核心 |
| 蒙特卡洛 | "对随机样本求平均" | 用样本平均近似积分。误差 O(1/sqrt(N))，与维度无关 |
| MCMC | "会收敛的随机游走" | 构造一条平稳分布等于目标分布的马尔可夫链。Metropolis-Hastings 是奠基算法 |
| Metropolis-Hastings | "上坡必接受，下坡偶尔接受" | 提议移动，按密度比决定接受与否。细致平衡保证收敛到目标分布 |
| 吉布斯采样 | "一次一个变量" | 固定其他变量，从每个变量的条件分布更新。接受率 100% |
| 温度 | "自信度旋钮" | 在 softmax 之前把 logits 除以 T。T<1 使分布变尖（更自信），T>1 使分布变平（更多样） |
| Top-k 采样 | "保留最好的 k 个" | 把除最高概率的 k 个词元以外的全部置零，重新归一化，再采样。候选集大小固定 |
| 核采样（top-p） | "保留概率高的那些" | 保留累积概率超过 p 的最小词元集合。候选集大小自适应 |
| 重参数化技巧 | "把随机性挪到外面" | 写成 z = mu + sigma * epsilon，其中 epsilon ~ N(0,1)。让采样可微。VAE 训练的关键 |
| Gumbel-Softmax | "软类别采样" | 用 Gumbel 噪声加带温度的 softmax 得到的类别采样可微近似 |
| 分层采样 | "强制覆盖" | 把样本空间分层，每层都采样。方差总低于朴素的蒙特卡洛 |
| 预烧期（burn-in） | "热身阶段" | MCMC 在链达到平稳分布之前丢弃的初始样本 |
| 细致平衡 | "可逆性条件" | p(x) * T(x->y) = p(y) * T(y->x)。p 成为马尔可夫链平稳分布的充分条件 |
| 扩散采样 | "迭代去噪" | 从噪声出发，应用学到的去噪步骤来生成数据。每一步都是一次条件采样操作 |

## 延伸阅读（Further Reading）

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - 关于 MCMC 基础的详细教程
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - Gumbel-Softmax 原始论文
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - 核（top-p）采样论文
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - 提出重参数化技巧的 VAE 论文
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM 把采样与图像生成联系起来
