import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def demo_svd_animation():

    # =========================================================
    # 1. 定义矩阵 A
    # =========================================================

    A = np.array([
        [3.0, 1.0],
        [1.0, 3.0]
    ])

    # SVD
    U, S, Vt = np.linalg.svd(A)

    print("A =")
    print(A)

    print("\nU =")
    print(np.round(U, 4))

    print("\nSigma =")
    print(np.round(S, 4))

    print("\nVt =")
    print(np.round(Vt, 4))

    print("\nCheck:")
    print(np.round(U @ np.diag(S) @ Vt, 4))

    # =========================================================
    # 2. 创建单位圆
    # =========================================================

    theta = np.linspace(0, 2 * np.pi, 300)

    circle = np.vstack([
        np.cos(theta),
        np.sin(theta)
    ])

    # =========================================================
    # 3. 三个阶段的目标
    # =========================================================

    # 阶段1：V^T
    circle_v = Vt @ circle

    # 阶段2：Sigma
    circle_sigma = np.diag(S) @ circle_v

    # 阶段3：U
    circle_final = U @ circle_sigma

    # =========================================================
    # 4. 创建画布
    # =========================================================

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    ax.set_aspect("equal")

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)

    ax.grid(True, alpha=0.3)

    ax.set_title("SVD Geometry")

    # =========================================================
    # 5. 创建圆
    # =========================================================

    line, = ax.plot([], [], linewidth=3)

    # 画出最终结果的淡淡轮廓
    final_line, = ax.plot(
        circle_final[0],
        circle_final[1],
        linestyle="--",
        alpha=0.3
    )

    # =========================================================
    # 6. 动画初始化
    # =========================================================

    def init():
        line.set_data([], [])
        return line, final_line

    # =========================================================
    # 7. 动画
    # =========================================================

    def update(frame):

        # ---------------------------------------------
        # frame 0 ~ 100
        # V^T：逐渐旋转
        # ---------------------------------------------

        if frame < 100:

            t = frame / 100

            # 从 I 逐渐变成 V^T
            M = (1 - t) * np.eye(2) + t * Vt

            points = M @ circle

            title = "Step 1: Vᵀ  → Rotate"

        # ---------------------------------------------
        # frame 100 ~ 200
        # Sigma：逐渐拉伸
        # ---------------------------------------------

        elif frame < 200:

            t = (frame - 100) / 100

            # 已经完成 V^T
            rotated = Vt @ circle

            # 从 1 倍逐渐变成 S
            scale = np.array([
                1 + t * (S[0] - 1),
                1 + t * (S[1] - 1)
            ])

            points = scale[:, None] * rotated

            title = "Step 2: Σ  → Scale"

        # ---------------------------------------------
        # frame 200 ~ 300
        # U：逐渐旋转
        # ---------------------------------------------

        else:

            t = (frame - 200) / 100

            rotated = Vt @ circle

            scaled = np.diag(S) @ rotated

            # U 是一个正交矩阵
            # 使用角度插值来实现旋转动画

            angle = np.arctan2(U[1, 0], U[0, 0])

            current_angle = t * angle

            R = np.array([
                [np.cos(current_angle), -np.sin(current_angle)],
                [np.sin(current_angle),  np.cos(current_angle)]
            ])

            points = R @ scaled

            title = "Step 3: U  → Rotate"

        # 更新圆/椭圆
        line.set_data(
            points[0],
            points[1]
        )

        ax.set_title(title)

        return line, final_line

    # =========================================================
    # 8. 创建动画
    # =========================================================

    animation = FuncAnimation(
        fig,
        update,
        frames=300,
        init_func=init,
        interval=30,
        blit=True
    )

    plt.show()


if __name__ == "__main__":
    demo_svd_animation()