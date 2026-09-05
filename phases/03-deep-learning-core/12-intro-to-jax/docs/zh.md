# JAX 入门（Introduction to JAX）

> PyTorch 修改张量，TensorFlow 构建图，JAX 编译纯函数。最后那一点会改变你对深度学习的思考方式。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 03 Lessons 01-10, basic NumPy
**Time:** ~90 minutes

## 学习目标（Learning Objectives）

- 使用 JAX 的函数式 API（jax.numpy、jax.grad、jax.jit、jax.vmap）编写纯函数式的神经网络代码
- 解释 PyTorch 的急切修改与 JAX 的函数式编译模型之间的关键设计差异
- 应用 jit 编译和 vmap 向量化，让训练循环比朴素的 Python 实现更快
- 在 JAX 中训练一个简单网络，并将其显式的状态管理与 PyTorch 的面向对象方式加以对比

## 问题所在（The Problem）

你知道怎么用 PyTorch 构建神经网络：定义一个 `nn.Module`，调用 `.backward()`，让优化器前进一步。它能用，几百万人都在用。

但 PyTorch 的基因里刻着一个约束：它在 Python 中急切地逐个追踪操作。每个 `tensor + tensor` 都是一次独立的内核启动，每个训练步骤都要重新解释同样的 Python 代码。这在一般情况下没问题，直到你需要用 2,048 块 TPU 训练一个 5400 亿参数的模型 —— 这时这些开销会把你拖垮。

Google DeepMind 用 JAX 训练 Gemini，Anthropic 用 JAX 训练 Claude。这些都不是小打小闹 —— 它们是地球上规模最大的神经网络训练任务。他们选择 JAX，是因为 JAX 把你的训练循环当作一个可编译的程序，而不是一串 Python 调用。

JAX 就是带三种超能力的 NumPy：自动微分、到 XLA 的 JIT 编译、自动向量化。你写一个处理单个样本的函数，JAX 给你的却是一个能处理整个批次、计算梯度、编译成机器码并跨多设备运行的函数。这一切都不需要改动原函数。

## 核心概念（The Concept）

### JAX 的哲学（The JAX Philosophy）

JAX 是一个函数式框架：没有类，没有可变状态，没有 `.backward()` 方法。取而代之的是：

| PyTorch | JAX |
|---------|-----|
| 带状态的 `nn.Module` 类 | 纯函数：`f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| 急切执行 | 通过 XLA 进行 JIT 编译 |
| `for x in batch:` 手动循环 | `jax.vmap(f)` 自动向量化 |
| `DataParallel` / `FSDP` | `jax.pmap(f)` 自动并行 |
| 可变的 `model.parameters()` | 不可变的数组 pytree |

这不是风格偏好，而是编译器的约束。JIT 编译要求纯函数（pure function）—— 相同的输入永远产生相同的输出，没有副作用。正是这条限制让 100 倍加速成为可能。

### jax.numpy：熟悉的接口（jax.numpy: The Familiar Surface）

JAX 在加速器上重新实现了 NumPy API：

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

函数名相同，广播规则相同，切片语义相同。但这些数组生活在 GPU/TPU 上，而且每个操作都能被编译器追踪。

一个关键差异：JAX 数组是不可变的。不能写 `a[0] = 5`，而要写 `a = a.at[0].set(5)`。这会让你别扭一个星期，然后豁然开朗 —— 正是不可变性让 `grad`、`jit` 和 `vmap` 这类变换可以自由组合。

### jax.grad：函数式自动微分（jax.grad: Functional Autodiff）

PyTorch 把梯度挂在张量上（`.grad`），JAX 把梯度挂在函数上。

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` 接收一个函数，返回一个计算梯度的新函数。没有 `.backward()` 调用，没有存在张量上的计算图。梯度就是另一个函数，你可以调用它、组合它，或者对它做 JIT 编译。

它可以任意组合：

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

二阶导数、三阶导数、雅可比矩阵（Jacobian）、海森矩阵（Hessian），全都靠组合 `grad` 得到。PyTorch 也能做到（`torch.autograd.functional.hessian`），但那是后加上去的；在 JAX 里，它是地基。

约束在于：`grad` 只对纯函数有效。函数内部不能有 print 语句（它们在追踪时运行，而不是在真正执行时），不能修改外部状态，没有显式的 key 管理就不能生成随机数。

### jit：编译到 XLA（jit: Compile to XLA）

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

第一次调用时，JAX 会对函数做追踪（trace）—— 记录会发生哪些操作，但不真正执行。然后它把这份踪迹交给 XLA（Accelerated Linear Algebra），也就是 Google 面向 TPU 和 GPU 的编译器。XLA 会融合操作、消除冗余的内存拷贝，并生成优化过的机器码。

之后的调用完全跳过 Python，编译后的代码以 C++ 的速度运行在加速器上。

JIT 什么时候有帮助：

- 训练步骤（同样的计算重复成千上万次）
- 推理（同一个模型，不同的输入）
- 任何会被以形状相近的输入反复调用的函数

JIT 什么时候帮倒忙：

- 带有依赖具体值的 Python 控制流的函数（`if x > 0`，其中 x 是被追踪的数组）
- 一次性计算（编译开销超过运行时间）
- 调试（追踪掩盖了实际执行过程）

控制流的限制是真实存在的。`jax.lax.cond` 取代 `if/else`，`jax.lax.scan` 取代 `for` 循环。这些不是可选项 —— 它们是编译的代价。

### vmap：自动向量化（vmap: Automatic Vectorization）

你写一个处理单个样本的函数：

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` 把它提升成能处理一个批次：

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` 的意思是：不对 `params` 分批（它被共享），对 `x` 的第 0 轴分批。不需要手写 `for` 循环，不需要变形，不需要手动穿针引线地传递批次维度。JAX 会自己搞定批次维度，并把整个计算向量化。

这不是语法糖。`vmap` 生成的融合向量化代码比 Python 循环快 10 到 100 倍。而且它还能与 `jit` 和 `grad` 组合：

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

逐样本梯度，一行代码。在 PyTorch 里不耍点花招几乎做不到。

### pmap：跨设备的数据并行（pmap: Data Parallelism Across Devices）

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` 把函数复制到所有可用设备（GPU/TPU）上，并切分批次。在函数内部，`jax.lax.pmean` 和 `jax.lax.psum` 负责跨设备同步梯度。

Google 用 `pmap`（以及它的后继者 `shard_map`）在数千块 TPU v5e 芯片上训练 Gemini。它的编程模型是：写好单设备版本，用 `pmap` 包一层，完事。

### Pytree：通用数据结构（Pytrees: The Universal Data Structure）

JAX 操作的是"pytree" —— 列表、元组、字典和数组的嵌套组合。你的模型参数就是一个 pytree：

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

每个 JAX 变换 —— `grad`、`jit`、`vmap` —— 都知道如何遍历 pytree。`jax.tree.map(f, tree)` 把 `f` 应用到每个叶子。优化器就是这样一次性更新所有参数的：

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

没有 `.parameters()` 方法，没有参数注册。树结构本身就是模型。

### 函数式 vs 面向对象（Functional vs Object-Oriented）

PyTorch 把状态存在对象里：

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX 使用显式传递状态的纯函数：

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

参数是传进来的。什么都不存储，什么都不修改。这让每个函数都可测试、可组合、可编译。这也意味着你要自己管理参数 —— 或者使用 Flax、Equinox 这类库。

### JAX 生态系统（The JAX Ecosystem）

JAX 给你原语，库给你好用的封装：

| 库 | 角色 | 风格 |
|---------|------|-------|
| **Flax**（Google） | 神经网络层 | 带显式状态的 `nn.Module` |
| **Equinox**（Patrick Kidger） | 神经网络层 | 基于 pytree，Python 风格 |
| **Optax**（DeepMind） | 优化器 + 学习率调度 | 可组合的梯度变换 |
| **Orbax**（Google） | 检查点（checkpoint） | 保存/恢复 pytree |
| **CLU**（Google） | 指标 + 日志 | 训练循环工具集 |

Optax 是标准的优化器库。它把梯度变换（Adam、SGD、裁剪）与参数更新分离开来，让组合变得轻而易举：

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### 什么时候用 JAX，什么时候用 PyTorch（When to Use JAX vs PyTorch）

| 因素 | JAX | PyTorch |
|--------|-----|---------|
| TPU 支持 | 一等公民（两者都是 Google 造的） | 社区维护（torch_xla） |
| GPU 支持 | 良好（通过 XLA 使用 CUDA） | 业界最佳（原生 CUDA） |
| 调试 | 困难（追踪 + 编译） | 简单（急切执行，逐行进行） |
| 生态系统 | 偏研究（Flax、Equinox） | 庞大（HuggingFace、torchvision 等） |
| 求职市场 | 小众（Google/DeepMind/Anthropic） | 主流（无处不在） |
| 大规模训练 | 更强（XLA、pmap、mesh） | 不错（FSDP、DeepSpeed） |
| 原型开发速度 | 较慢（函数式开销） | 较快（改完就跑） |
| 生产推理 | TensorFlow Serving、Vertex AI | TorchServe、Triton、ONNX |
| 谁在用 | DeepMind（Gemini）、Anthropic（Claude） | Meta（Llama）、OpenAI（GPT）、Stability AI |

老实的答案是：除非你有使用 JAX 的特定理由，否则就用 PyTorch。这些理由是 —— 能用上 TPU、需要逐样本梯度、超大规模的多设备训练，或者在 Google/DeepMind/Anthropic 工作。

### JAX 中的随机数（Random Numbers in JAX）

JAX 没有全局随机状态。每个随机操作都需要一个显式的 PRNG key：

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

一开始会觉得麻烦。但它保证了跨设备、跨编译的可复现性 —— 这是 PyTorch 的 `torch.manual_seed` 在多 GPU 环境下无法保证的性质。

```figure
batchnorm-effect
```

## 动手构建（Build It）

### 第 1 步：准备环境与数据（Step 1: Setup and Data）

我们将用 JAX 和 Optax 在 MNIST 上训练一个 3 层 MLP：784 个输入，两个隐藏层分别有 256 和 128 个神经元，10 个输出类别。

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### 第 2 步：初始化参数（Step 2: Initialize Parameters）

没有类，只有一个返回 pytree 的函数：

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

手动实现的 He 初始化。三个 PRNG key 从一个种子分裂而来。每个权重都是嵌套字典里的一个不可变数组。

### 第 3 步：前向传播（Step 3: Forward Pass）

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

纯函数：参数进去，预测出来。没有 `self`，没有存储的状态。`loss_fn` 从零开始计算交叉熵 —— softmax、取对数、取负的平均。

### 第 4 步：JIT 编译的训练步骤（Step 4: JIT-Compiled Training Step）

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` 一次前向同时返回损失值和梯度。`@jax.jit` 装饰器把这两个函数都编译到 XLA。第一次调用之后，每个训练步骤都不再经过 Python。

### 第 5 步：训练循环（Step 5: Training Loop）

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 个 epoch，约 97% 的测试准确率。第一个 epoch 很慢（JIT 编译），第 2 到 10 个 epoch 很快。

注意这里少了什么：没有 `.zero_grad()`，没有 `.backward()`，没有 `.step()`。整个更新就是一次组合起来的函数调用。梯度被计算出来、经过 Adam 变换、再应用到参数上 —— 全部发生在 `train_step` 里面。

## 使用它（Use It）

### Flax：Google 的标准（Flax: The Google Standard）

Flax 是最常用的 JAX 神经网络库。它把 `nn.Module` 加了回来，但状态是显式管理的：

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

结构与 PyTorch 相同，但 `params` 与模型是分离的。`model.init()` 创建参数，`model.apply(params, x)` 执行前向传播。模型对象本身没有状态。

### Equinox：更 Pythonic 的选择（Equinox: The Pythonic Alternative）

Equinox（Patrick Kidger 开发）把模型表示成 pytree：

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

模型本身就是 pytree，不需要 `.apply()`。参数就是模型的叶子。这更接近 JAX 的思维方式。

### Optax：可组合的优化器（Optax: Composable Optimizers）

Optax 把梯度变换与参数更新解耦：

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

梯度裁剪、学习率预热（warmup）、权重衰减 —— 全部组合成一条变换链。每个变换接收梯度、修改梯度，再交给下一个。没有大而全的单一优化器类。

## 交付（Ship It）

**安装：**

```bash
pip install jax jaxlib optax flax
```

如果需要 GPU 支持：

```bash
pip install jax[cuda12]
```

如果需要 TPU（Google Cloud）：

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**性能陷阱：**

- 第一次 JIT 调用很慢（要编译）。基准测试前先预热。
- 在 JIT 内部避免用 Python 循环遍历 JAX 数组。改用 `jax.lax.scan` 或 `jax.lax.fori_loop`。
- `jax.debug.print()` 在 JIT 内部可用，普通的 `print()` 不行。
- 用 `jax.profiler` 或 TensorBoard 做性能分析。XLA 编译可能掩盖真正的瓶颈。
- JAX 默认预分配 75% 的 GPU 显存。设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 可关闭这一行为。

**检查点（checkpoint）：**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**本课产出：**
- `outputs/prompt-jax-optimizer.md` —— 一个用于选择合适 JAX 优化器配置的提示词
- `outputs/skill-jax-patterns.md` —— 一份涵盖 JAX 函数式模式的技能

## 练习（Exercises）

1. 给 MLP 加上 dropout。在 JAX 里，dropout 需要一个 PRNG key —— 把 key 穿插进前向传播，并为每个 dropout 层分裂出子 key。比较开启与不开启 dropout 的测试准确率。

2. 用 `jax.vmap` 计算一个批次（32 张 MNIST 图像）的逐样本梯度。算出每个样本的梯度范数。哪些样本的梯度最大？为什么？

3. 把手写的 forward 函数换成一个适用于任意层数的通用 `mlp_forward(params, x)`。用 `jax.tree.leaves` 自动判断深度。

4. 对比有无 `@jax.jit` 时的训练步骤性能。分别给 100 步计时。在你的硬件上加速有多大？第一次调用的编译开销是多少？

5. 通过组合 `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))` 实现梯度裁剪。分别在开启与不开启裁剪的情况下训练，并画出训练过程中梯度范数的变化，观察效果。

## 关键术语（Key Terms）

| 术语 | 人们常说什么 | 实际含义 |
|------|----------------|----------------------|
| XLA | "让 JAX 变快的东西" | Accelerated Linear Algebra —— 一个从计算图出发、融合操作并生成优化 GPU/TPU 内核的编译器 |
| JIT | "即时编译" | JAX 在第一次调用时追踪函数、编译到 XLA，之后的调用直接运行编译后的版本 |
| 纯函数（Pure function） | "没有副作用" | 输出只取决于输入的函数 —— 没有全局状态，没有修改，没有显式 key 就没有随机性 |
| vmap | "自动批处理" | 把处理单个样本的函数变换成处理一个批次的函数，无需重写 |
| pmap | "自动并行" | 把函数复制到多个设备上，并切分输入批次 |
| Pytree | "嵌套的数组字典" | 任何由列表、元组、字典和数组构成的嵌套结构，JAX 可以遍历并变换它 |
| 追踪（Tracing） | "记录计算过程" | JAX 用抽象值执行函数以构建计算图，但不计算真实结果 |
| 函数式自动微分（Functional autodiff） | "对函数求 grad" | 通过变换函数来求导数，而不是把梯度存储挂在张量上 |
| Optax | "JAX 的优化器库" | 一个可组合的梯度变换库 —— Adam、SGD、裁剪、调度 —— 可以链式串联 |
| Flax | "JAX 的 nn.Module" | Google 面向 JAX 的神经网络库，在保持状态显式的前提下增加了层抽象 |

## 延伸阅读（Further Reading）

- JAX 文档：https://jax.readthedocs.io/ —— 官方文档，里面有关于 grad、jit 和 vmap 的优秀教程
- "JAX: composable transformations of Python+NumPy programs" (Bradbury et al., 2018) —— 解释设计哲学的原始论文
- Flax 文档：https://flax.readthedocs.io/ —— Google 面向 JAX 的神经网络库
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations" (2021) —— 比 Flax 更 Pythonic 的替代品
- DeepMind, "Optax: composable gradient transformation and optimisation" —— 标准的优化器库
- "You Don't Know JAX" (Colin Raffel, 2020) —— 一本关于 JAX 陷阱与模式的实用指南，出自 T5 作者之一
