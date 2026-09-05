# 数值稳定性（Numerical Stability）

> 浮点数是一个有漏洞的抽象。它会在训练中咬你一口，而且你根本预见不到。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~120 minutes

## 学习目标（Learning Objectives）

- 使用减最大值（max-subtraction）技巧实现数值稳定的 softmax 和 log-sum-exp
- 识别浮点计算中的上溢、下溢和灾难性抵消
- 使用中心有限差分，用数值梯度验证解析梯度
- 解释训练中为什么首选 bfloat16 而非 float16，以及损失缩放（loss scaling）如何防止梯度下溢

## 问题（The Problem）

你的模型训练了三个小时，然后 loss 变成了 NaN。你加了一行 print。第 9,000 步 logits 还正常，第 9,001 步就变成了 `inf`，到第 9,002 步所有梯度都是 `nan`，训练宣告死亡。

又或者：你的模型完整跑完了训练，但准确率比论文宣称的低 2%。你检查了一切：架构一致，超参数一致，数据一致。问题在于论文用的是 float32，而你用的是 float16 且没做正确的缩放。32 位上累积的舍入误差悄悄吃掉了你的准确率。

再或者：你从零实现了交叉熵损失。logits 较小时一切正常，一旦超过 100 就返回 `inf`。softmax 上溢了，因为 `exp(100)` 超出了 float32 能表示的范围。每个机器学习框架都用一个两行的小技巧处理这个问题，而你根本不知道有这个技巧。

数值稳定性不是什么理论层面的顾虑。它决定了训练是成功还是悄无声息地失败。你将来调试的每一个严重的机器学习 bug，最终都会归结到浮点数上。

## 概念（The Concept）

### IEEE 754：计算机如何存储实数（IEEE 754: How Computers Store Real Numbers）

计算机按照 IEEE 754 标准用浮点值存储实数。一个浮点数由三部分组成：符号位（sign bit）、指数（exponent）和尾数（mantissa，即有效数字 significand）。

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

尾数决定精度（有多少位有效数字），指数决定范围（数能有多大或多小）。

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 提供大约 7 位十进制精度。它能区分 1.0000001 和 1.0000002，却分不清 1.00000001 和 1.00000002。超过 7 位之后，一切都是舍入噪声。

float16 只有大约 3 位精度，能表示的最大数是 65,504。对机器学习来说这个上限小得令人不安——logits、梯度和激活值动辄就超过它。

bfloat16 是 Google 对 float16 范围问题的回应。它与 float32 使用相同的 8 位指数（同样的范围，最大到 3.4e38），但尾数只有 7 位（精度低于 float16）。对训练神经网络而言，范围比精度更重要，所以 bfloat16 通常更胜一筹。

### 为什么 0.1 + 0.2 != 0.3（Why 0.1 + 0.2 != 0.3）

0.1 这个数无法在二进制浮点数中被精确表示。以 2 为底时，它是一个无限循环小数：

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

float32 把它截断为 23 位尾数，存储的值约为 0.100000001490116。类似地，0.2 被存成约 0.200000002980232。两者之和是 0.300000004470348，而不是 0.3。

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这对机器学习很重要，因为：

1. 像 `if loss < threshold` 这样的 loss 比较可能给出错误答案
2. 累积大量小数值（数千步的梯度更新）会偏离真实总和
3. 如果用 `==` 比较浮点数，校验和与可复现性测试都会失败

解决办法：永远不要用 `==` 比较浮点数，改用 `abs(a - b) < epsilon` 或 `math.isclose()`。

### 灾难性抵消（Catastrophic Cancellation）

当你把两个几乎相等的浮点数相减时，有效数字相互抵消，剩下的舍入噪声被顶到了最高有效位。

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

一次减法就带来 19% 的相对误差。在机器学习中，只要出现以下情况就会遇到它：

- 对均值很大的数据计算方差：E[x] 很大时使用 `E[x^2] - E[x]^2`
- 对几乎相等的对数概率做减法
- 用太小的 epsilon 计算有限差分梯度

解决办法：改写公式，避免对两个大而几乎相等的数做减法。算方差时，使用 Welford 算法或先对数据做中心化。处理对数概率时，全程在对数空间中运算。

### 上溢与下溢（Overflow and Underflow）

上溢（overflow）指结果大到无法表示。下溢（underflow）指结果小到无法表示（比最小可表示的正数还要接近零）。

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

`exp()` 函数是机器学习中上溢的主要来源：

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

`log()` 函数的问题则出在另一个方向：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

在机器学习中，`exp()` 出现在 softmax、sigmoid 和概率计算里，`log()` 出现在交叉熵、对数似然和 KL 散度里。没有正确的技巧，`log(exp(x))` 这样的组合就是一片雷区。

### Log-Sum-Exp 技巧（The Log-Sum-Exp Trick）

直接计算 `log(sum(exp(x_i)))` 在数值上非常危险。只要有某个 `x_i` 很大，`exp(x_i)` 就会上溢；如果所有 `x_i` 都非常负，每个 `exp(x_i)` 都会下溢成零，`log(0)` 等于 `-inf`。

技巧：在求指数之前先减去最大值。

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

为什么可行：减去 `max(x)` 之后，最大的指数是 `exp(0) = 1`，不可能上溢；求和项中至少有一项是 1，所以总和至少是 1，而 `log(1) = 0`，也不可能下溢到 `-inf`。

证明：

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

令 `c = max(x)`，上溢就被消除了。

这个技巧在机器学习中无处不在：
- softmax 归一化
- 交叉熵损失计算
- 序列模型中的对数概率求和
- 高斯混合模型
- 变分推断

### 为什么 softmax 需要减最大值技巧（Why Softmax Needs the Max-Subtraction Trick）

softmax 把 logits 转换成概率：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

不加技巧的话，[100, 101, 102] 这样的 logits 会导致上溢：

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

用了技巧，减去 max(x) = 102：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

概率结果完全相同，但计算是安全的。这不是优化，而是正确性的必要条件。

### NaN 与 Inf：检测与预防（NaN and Inf: Detection and Prevention）

`nan`（Not a Number，非数）和 `inf`（无穷大）会在计算中像病毒一样传播。梯度更新中出现一个 `nan`，权重就变成 `nan`，随后每个输出都变成 `nan`。训练一步之内就宣告死亡。

`inf` 的产生途径：
- 对很大的正数做 `exp()`
- 除以零：`1.0 / 0.0`
- 累加过程中的 `float32` 上溢

`nan` 的产生途径：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 对负数做 `sqrt()`
- 对负数做 `log()`
- 任何涉及已有 `nan` 的运算

检测：

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

预防策略：

1. 对 `exp()` 的输入做截断：`exp(clamp(x, -80, 80))`
2. 给分母加 epsilon：`x / (y + 1e-8)`
3. 在 `log()` 内部加 epsilon：`log(x + 1e-8)`
4. 使用稳定的实现（log-sum-exp、稳定版 softmax）
5. 梯度裁剪，防止权重爆炸
6. 调试期间在每次前向传播后检查 `nan`/`inf`

### 数值梯度检查（Numerical Gradient Checking）

解析梯度（来自反向传播）可能有 bug。数值梯度检查通过有限差分计算梯度来验证它们。

中心差分公式：

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

它的精度是 O(h^2)，远好于只有 O(h) 精度的前向差分 `(f(x+h) - f(x)) / h`。

选择 h：太大会让近似失真，太小则灾难性抵消会毁掉结果。通常取 `h = 1e-5` 到 `1e-7`。

检查方法：计算解析梯度与数值梯度之间的相对差。

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验法则：
- relative_error < 1e-7：完美，梯度正确
- relative_error < 1e-5：可接受，大概率正确
- relative_error > 1e-3：有问题
- relative_error > 1：梯度完全错了

实现新的层或损失函数时，一定要检查梯度。PyTorch 提供了 `torch.autograd.gradcheck()` 来做这件事。

### 混合精度训练（Mixed Precision Training）

现代 GPU 配备专用硬件（Tensor Cores），计算 float16 矩阵乘法的速度可达 float32 的 2-8 倍。混合精度训练正是利用了这一点：

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

纯 float16 训练的问题在于：梯度往往非常小（1e-8 或更小），而 float16 会把低于约 6e-8 的值下溢成零。所有梯度更新都是零，模型就此停止学习。

解决办法是损失缩放（loss scaling）：

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

动态损失缩放会自动调整缩放因子：从一个较大的值（65536）开始；如果梯度上溢成 `inf`，就减半；如果连续 N 步没有上溢，就翻倍。

### bfloat16 对比 float16：为什么训练时 bfloat16 胜出（bfloat16 vs float16: Why bfloat16 Wins for Training）

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16 精度更高（尾数 10 位对 7 位），但范围有限（最大约 65,504）。bfloat16 精度较低，但范围与 float32 相同（最大约 3.4e38）。

训练神经网络时：

- 训练高峰期激活值和 logits 经常超过 65,504。float16 会上溢，bfloat16 能扛住。
- float16 需要损失缩放，bfloat16 通常不需要，因为它的范围覆盖了整个梯度量级谱。
- bfloat16 只是对 float32 的简单截断：丢掉尾数的低 16 位。转换非常简单，且指数部分无损。

推理时数值有界、精度更重要，所以首选 float16。训练时范围更重要，所以首选 bfloat16。这就是 TPU 和现代 NVIDIA GPU（A100、H100）都原生支持 bfloat16 的原因。

### 梯度裁剪（Gradient Clipping）

当梯度在多层之间指数级增长时就会发生梯度爆炸（常见于 RNN、深层网络和 transformer）。一个大梯度就能在一步之内毁掉所有权重。

裁剪分两种：

**按值裁剪（clip by value）：**独立截断每个梯度元素。

```
grad = clamp(grad, -max_val, max_val)
```

简单，但可能改变梯度向量的方向。

**按范数裁剪（clip by norm）：**缩放整个梯度向量，使其范数不超过阈值。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

保留梯度的方向。`torch.nn.utils.clip_grad_norm_()` 做的就是这件事，它是标准选择。

典型取值：transformer 用 `max_norm=1.0`，强化学习用 `max_norm=0.5`，较简单的网络用 `max_norm=5.0`。

梯度裁剪不是投机取巧，而是一种安全机制。没有它，一个异常批次产生的大梯度就足以毁掉几周的训练成果。

### 作为数值稳定器的归一化层（Normalization Layers as Numerical Stabilizers）

批归一化（batch normalization）、层归一化（layer normalization）和 RMS 归一化通常被介绍成帮助训练收敛的正则化手段，但它们同时也是数值稳定器。

没有归一化的话，激活值会随着层数指数级增大或缩小：

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

归一化在每一层对激活值重新居中并重新缩放：

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`（通常为 1e-5）在所有激活值完全相同时防止除以零。可学习参数 `gamma` 和 `beta` 让网络能够恢复所需的任意尺度。

这让数值在整个网络中都保持在安全范围内，同时防止前向传播中的上溢和反向传播中的梯度爆炸。

### 常见的机器学习数值 bug（Common ML Numerical Bugs）

**Bug：训练几个 epoch 后 loss 变成 NaN。**
原因：logits 增长得太大，softmax 上溢；或者学习率过高导致权重发散。
修复：使用稳定版 softmax（减最大值）、降低学习率、加上梯度裁剪。

**Bug：loss 卡在 log(num_classes) 不动。**
原因：模型输出接近均匀分布的概率。这通常意味着梯度消失，或模型根本没有在学习。
修复：检查数据标签是否正确、核对损失函数、检查是否有死掉的 ReLU。

**Bug：验证准确率比预期低 1-3%。**
原因：使用了混合精度却没有正确的损失缩放。梯度下溢悄悄把小更新变成零。
修复：启用动态损失缩放，或改用 bfloat16。

**Bug：某些层的梯度范数为 0.0。**
原因：ReLU 神经元死亡（输入全为负），或 float16 下溢。
修复：改用 LeakyReLU 或 GELU、使用梯度缩放、检查权重初始化。

**Bug：模型在一块 GPU 上正常，在另一块上结果不同。**
原因：浮点累加顺序不确定。GPU 并行归约在不同硬件上以不同顺序求和，而浮点加法不满足结合律。
修复：接受微小差异（1e-6），或者设置 `torch.use_deterministic_algorithms(True)` 并接受速度损失。

**Bug：loss 计算中 `exp()` 返回 `inf`。**
原因：没有使用减最大值技巧，直接把原始 logits 传给了 `exp()`。
修复：改用 `torch.nn.functional.log_softmax()`，它内部实现了 log-sum-exp。

**Bug：从 float32 切换到 float16 后训练发散。**
原因：float16 无法表示低于 6e-8 的梯度量级，也无法表示高于 65,504 的激活值。
修复：使用带损失缩放的混合精度（AMP），或改用 bfloat16。

```figure
logsumexp-stability
```

## 动手构建（Build It）

### 第 1 步：演示浮点精度极限（Step 1: Demonstrate floating point precision limits）

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### 第 2 步：实现朴素版与稳定版 softmax（Step 2: Implement naive vs stable softmax）

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### 第 3 步：实现稳定的 log-sum-exp（Step 3: Implement stable log-sum-exp）

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### 第 4 步：实现稳定的交叉熵（Step 4: Implement stable cross-entropy）

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### 第 5 步：梯度检查（Step 5: Gradient checking）

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 使用它（Use It）

### 混合精度模拟（Mixed precision simulation）

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 梯度裁剪（Gradient clipping）

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf 检测（NaN/Inf detection）

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

完整实现及所有边界情况的演示见 `code/numerical.py`。

## 交付（Ship It）

本课产出：
- `code/numerical.py`：包含稳定的 softmax、log-sum-exp、交叉熵、梯度检查和混合精度模拟
- `outputs/prompt-numerical-debugger.md`：用于诊断训练中的 NaN/Inf 和数值问题

这些稳定实现会在第 3 阶段构建训练循环时、第 4 阶段实现注意力机制时再次派上用场。

## 练习（Exercises）

1. **灾难性抵消。** 用朴素公式 `E[x^2] - E[x]^2` 在 float32 下计算 [1000000.0, 1000001.0, 1000002.0] 的方差，然后用 Welford 在线算法再算一次。将两种误差与真实方差（0.6667）对比。

2. **精度搜寻。** 在 Python 中找到使 `1.0 + x == 1.0` 成立的最小正 float32 值 `x`。它就是机器精度（machine epsilon）。验证它与 `numpy.finfo(numpy.float32).eps` 一致。

3. **Log-sum-exp 边界情况。** 用以下输入测试你的 `logsumexp_stable` 函数：(a) 所有值相等，(b) 有一个值远大于其余，(c) 所有值都非常负（-1000）。验证在朴素版本失败的地方它能给出正确结果。

4. **对神经网络层做梯度检查。** 实现单个线性层 `y = Wx + b` 及其解析反向传播，用 `numerical_gradient` 验证一个 3x2 权重矩阵的正确性。

5. **损失缩放实验。** 模拟 float16 训练：生成范围在 [1e-9, 1e-3] 的随机梯度，转换成 float16，统计变成零的比例。然后做损失缩放（乘以 1024），转换成 float16，再缩放回去，重新统计变成零的比例。

## 关键术语（Key Terms）

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| IEEE 754 | "浮点数标准" | 定义二进制浮点格式、舍入规则和特殊值（inf、nan）的国际标准。每块现代 CPU 和 GPU 都实现了它。 |
| 机器精度（machine epsilon） | "精度极限" | 在给定浮点格式下使 1.0 + e != 1.0 的最小值 e。float32 约为 1.19e-7。 |
| 灾难性抵消（catastrophic cancellation） | "减法带来的精度损失" | 两个几乎相等的浮点数相减时，有效数字相互抵消，舍入噪声主导了结果。 |
| 上溢（overflow） | "数太大了" | 结果超过最大可表示值，变成 inf。exp(89) 会让 float32 上溢。 |
| 下溢（underflow） | "数太小了" | 结果比最小可表示正数更接近零，变成 0.0。exp(-104) 会让 float32 下溢。 |
| Log-sum-exp 技巧 | "先减去最大值" | 通过提出 exp(max(x)) 来计算 log(sum(exp(x)))，防止上溢和下溢。用于 softmax、交叉熵和对数概率运算。 |
| 稳定 softmax | "不会爆炸的 softmax" | 在求指数前先减去 max(logits)。数值结果完全相同，且不可能上溢。 |
| 梯度检查（gradient checking） | "验证你的反向传播" | 把反向传播得到的解析梯度与有限差分得到的数值梯度做比较，以发现实现 bug。 |
| 混合精度（mixed precision） | "float16 前向，float32 反向" | 对速度关键的操作使用低精度浮点数，对数值敏感的操作使用高精度浮点数。典型加速比为 2-3 倍。 |
| 损失缩放（loss scaling） | "防止梯度下溢" | 反向传播前把 loss 乘以一个大常数，使梯度保持在 float16 的可表示范围内，然后在权重更新前除以同一个常数。 |
| bfloat16 | "Brain 浮点数" | Google 的 16 位格式：8 位指数（与 float32 范围相同），7 位尾数（精度低于 float16）。训练时的首选。 |
| 梯度裁剪（gradient clipping） | "给梯度范数设上限" | 缩放梯度向量使其范数不超过阈值，防止梯度爆炸毁掉权重。 |
| NaN | "Not a Number（非数）" | 由未定义运算（0/0、inf-inf、sqrt(-1)）产生的特殊浮点值，会传播到后续所有运算中。 |
| Inf | "无穷大" | 由上溢或除以零产生的特殊浮点值。相互组合可产生 NaN（inf - inf、inf * 0）。 |
| 数值梯度（numerical gradient） | "暴力求导" | 通过计算 f(x+h) 和 f(x-h) 再除以 2h 来近似导数。慢，但用于验证非常可靠。 |

## 延伸阅读（Further Reading）

- [每位计算机科学家都应了解的浮点数运算（Goldberg 1991）](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)——权威参考，内容密集但完备
- [混合精度训练（Micikevicius 等，2018）](https://arxiv.org/abs/1710.03740)——NVIDIA 提出 float16 训练损失缩放的论文
- [AMP：自动混合精度（PyTorch 文档）](https://pytorch.org/docs/stable/amp.html)——PyTorch 混合精度实用指南
- [bfloat16 格式（Google Cloud TPU 文档）](https://cloud.google.com/tpu/docs/bfloat16)——Google 为什么为 TPU 选择这种格式
- [Kahan 求和（Wikipedia）](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)——减少浮点求和舍入误差的算法
