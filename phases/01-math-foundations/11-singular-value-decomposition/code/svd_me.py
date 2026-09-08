"""用幂迭代实现简化版 SVD，并压缩 1.jpg。"""

from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# def power_iteration(M, num_iters=100):
#     """估算 M 的最大特征值和特征向量。"""
#     v = np.random.randn(M.shape[1])
#     v /= np.linalg.norm(v)

#     for _ in range(num_iters):
#         Mv = M @ v
#         norm = np.linalg.norm(Mv)
#         if norm == 0:
#             break
#         v = Mv / norm

#     return v @ M @ v, v
def power_iteration(M, num_iters=100):
    n = M.shape[1]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    for _ in range(num_iters):
        Mv = M @ v
        v = Mv / np.linalg.norm(Mv)

    eigenvalue = v @ M @ v
    return eigenvalue, v

def svd_from_scratch(A, k=None):
    """用幂迭代逐个提取 SVD 成分。"""
    m, n = A.shape
    k = min(m, n) if k is None else k
    sigmas, us, vs = [], [], []
    residual = A.copy().astype(float)

    for _ in range(k):
        eigenvalue, v = power_iteration(residual.T @ residual, 200)
        if eigenvalue < 1e-10:
            break

        sigma = np.sqrt(eigenvalue)
        u = residual @ v / sigma
        u /= np.linalg.norm(u)
        sigmas.append(sigma)
        us.append(u)
        vs.append(v)

        # 移除当前的秩 1 成分。
        residual -= sigma * np.outer(u, v)

    U = np.column_stack(us) if us else np.empty((m, 0))
    S = np.array(sigmas)
    V = np.column_stack(vs) if vs else np.empty((n, 0))
    return U, S, V


def compress_channel(channel, k):
    """压缩单个灰度通道。"""
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]


def compress_image(image, k):
    """对 RGB 三个通道分别做截断 SVD。"""
    channels = [compress_channel(image[:, :, i], k) for i in range(3)]
    return np.clip(np.stack(channels, axis=2), 0, 255).astype(np.uint8)


def image_demo():
    image_path = Path(__file__).with_name("1.jpg")
    output_path = Path(__file__).parents[1] / "outputs" / "svd_image_compression.png"
    output_path.parent.mkdir(exist_ok=True)

    # 缩小演示图，避免对原始大图做过慢的分解。
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((600, 600))
    original = np.asarray(image, dtype=float)
    height, width = original.shape[:2]
    ranks = [5, 20, 50, 100]
    ranks = [k for k in ranks if k < min(height, width)]

    compressed_images = [compress_image(original, k) for k in ranks]
    original_size = height * width * 3

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes[0, 0].imshow(original.astype(np.uint8))
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    for ax, k, compressed in zip(axes.flat[1:], ranks, compressed_images):
        error = np.linalg.norm(original - compressed) / np.linalg.norm(original)
        storage = k * (height + width + 1) * 3
        ratio = storage / original_size
        ax.imshow(compressed)
        ax.set_title(f"rank={k} | error={error:.3f} | storage={ratio:.1%}")
        ax.axis("off")

    # 没有填满的子图隐藏掉。
    for ax in axes.flat[1 + len(ranks):]:
        ax.axis("off")

    fig.suptitle(f"SVD image compression: {width} x {height}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)

    print(f"Input:  {image_path}")
    print(f"Output: {output_path}")
    for k, compressed in zip(ranks, compressed_images):
        error = np.linalg.norm(original - compressed) / np.linalg.norm(original)
        storage = k * (height + width + 1) * 3 / original_size
        print(f"rank={k:>3}  error={error:.4f}  storage={storage:.1%}")

def compress_image_svd(image_matrix, k):
    """使用自定义 SVD 函数压缩图像。"""
    U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)
    compressed =  U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :] # 保留前k个
    return compressed

def compress_dome():
    image  = np.random.seed(42)
    rows, cols = 200, 300
    image = np.random.rand(rows, cols)

    for i in [1, 5, 10, 20, 50]:
        compress_image = compress_image_svd(image, i)
        # 计算误差 这里用到了前面信息熵 $$E=A-A_k$$
        error = np.linalg.norm(image - compress_image) / np.linalg.norm(image) # np.linalg.norm 计算向量或矩阵的范数(所有元素平方和再开方)
        original_size = rows * cols # 计算存贮空间
        compress_size = i * (rows + cols + 1)
        ratio  = compress_size / original_size
        print(f"rank={i:>3}  error={error:.4f}  storage={ratio:.1%}")
        print(f"compressed shape: {compress_image.shape}")


def denoise_image_svd(noisy_image, k):
    np.random.seed(42)
    # SVD实现噪声去除 outer 向量外积
    clean = np.outer(np.sin(np.linspace(0, 4*np.pi, 100)),#等差数列生成器
                    np.cos(np.linspace(0, 2*np.pi, 80)))
    noise = clean + 0.5 * np.random.randn(100, 80)
    noisy = clean + noise

    U, S, Vt = np.linalg.svd(noisy, full_matrices=False)
    denoised = U[:, :5] @ np.diag(S[:5]) @ Vt[:5, :]

    print(f"Noisy error:    {np.linalg.norm(noisy - clean):.4f}")
    print(f"Denoised error: {np.linalg.norm(denoised - clean):.4f}")
    print(f"Improvement:    {(1 - np.linalg.norm(denoised - clean) / np.linalg.norm(noisy - clean)):.1%}")

def pseudoinverse_example():
    # 伪逆
    A = np.array([[1, 1], [2, 1], [3, 1]], dtype=float)
    b = np.array([3, 5, 6], dtype=float)

    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S_inv = np.diag(1.0 / S)
    A_pinv = Vt.T @ S_inv @ U.T

    x_svd = A_pinv @ b
    x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
    x_pinv = np.linalg.pinv(A) @ b

    print(f"SVD pseudoinverse solution:  {x_svd}")
    print(f"np.linalg.lstsq solution:   {x_lstsq}")
    print(f"np.linalg.pinv solution:    {x_pinv}")