class Vector:
    def __init__(self, components):
        """
        创建一个向量对象。

        参数:
            components (iterable): 向量分量的可迭代对象，例如列表或元组。

        示例:
            >>> v = Vector([1, 2, 3])
            >>> v.components
            [1, 2, 3]
        """
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        """
        向量逐元素相加（使用 zip，因此长度以较短者为准截断）。

        参数:
            other (Vector): 另一个向量。

        返回:
            Vector: 相加得到的新向量。

        示例:
            >>> Vector([1,2]) + Vector([3,4])
            Vector([4, 6])
        """
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        """
        向量逐元素相减（使用 zip，长度以较短者为准）。

        示例:
            >>> Vector([5,6]) - Vector([2,1])
            Vector([3, 5])
        """
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def __mul__(self, scalar):
        """
        标量乘法：向量的每个分量乘以标量。

        参数:
            scalar (number): 标量值。

        返回:
            Vector: 结果向量。

        示例:
            >>> Vector([1,2]) * 3
            Vector([3, 6])
        """
        return Vector([x * scalar for x in self.components])

    def dot(self, other):
        """
        计算点积（内积）。如果向量长度不一致，使用 zip 截断到较短长度。

        参数:
            other (Vector): 另一个向量。

        返回:
            number: 点积值。

        示例:
            >>> Vector([1,2,3]).dot(Vector([4,5,6]))
            32
        """
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        """
        计算向量的范数（欧几里得长度）。

        返回:
            float: 向量的长度。

        示例:
            >>> Vector([3,4]).magnitude()
            5.0
        """
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        """
        返回归一化后的单位向量（长度为 1）。

        注意: 如果向量为零向量，当前实现会发生除零错误（ZeroDivisionError）。
        在使用前应检查 `magnitude()` 是否为 0。

        示例:
            >>> Vector([3,4]).normalize()
            Vector([0.6, 0.8])
        """
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        """
        计算余弦相似度：cos(θ) = (a·b) / (|a||b|)。

        注意: 如果任一向量为零向量，会发生除零错误。

        示例:
            >>> Vector([1,0]).cosine_similarity(Vector([0,1]))
            0.0
        """
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def angle_between(self, other):
        """
        计算两向量之间的夹角（以度为单位）。内部先计算余弦相似度并限制在 [-1,1] 以避免数值问题。

        示例:
            >>> Vector([1,0]).angle_between(Vector([0,1]))
            90.0
        """
        import math
        cos_theta = self.cosine_similarity(other)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        return math.degrees(math.acos(cos_theta))

    def project_onto(self, other):
        """
        将当前向量投影到另一个向量上（正交投影）。

        公式: proj_u(v) = (v·u / u·u) * u

        注意: 如果 `other` 为零向量，会发生除零错误。

        示例:
            >>> Vector([3,4]).project_onto(Vector([1,0]))
            Vector([3.0, 0.0])
        """
        scalar = self.dot(other) / other.dot(other)
        return Vector([scalar * x for x in other.components])

    def __repr__(self):
        return f"Vector({self.components})"


def is_independent(vectors):
    n = len(vectors)
    if n == 0:
        return True
    dim = vectors[0].dim
    rows = [v.components[:] for v in vectors]
    rank = 0
    for col in range(dim):
        pivot = None
        for row in range(rank, len(rows)):
            if abs(rows[row][col]) > 1e-10:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [x / scale for x in rows[rank]]
        for row in range(len(rows)):
            if row != rank and abs(rows[row][col]) > 1e-10:
                factor = rows[row][col]
                rows[row] = [rows[row][j] - factor * rows[rank][j] for j in range(dim)]
        rank += 1
    return rank == n


"""
检查向量组是否线性无关。

参数:
    vectors (list[Vector]): 向量列表，将每个向量视为行向量进行秩计算。

返回:
    bool: 若向量组线性无关返回 True，否则 False。

示例:
    >>> e1 = Vector([1,0,0])
    >>> e2 = Vector([0,1,0])
    >>> e3 = Vector([0,0,1])
    >>> is_independent([e1,e2,e3])
    True
"""


def gram_schmidt(vectors):
    """
    对一组向量应用 Gram–Schmidt 正交化，返回一个正交归一基（orthonormal basis）。

    注意: 函数假设 `orthonormal` 内元素为向量，返回的向量是归一化过的单位向量。

    示例:
        >>> u1 = Vector([1,1,0])
        >>> u2 = Vector([1,0,1])
        >>> basis = gram_schmidt([u1, u2])
        >>> len(basis) >= 1
        True
    """
    orthonormal = []
    for v in vectors:
        w = v
        for u in orthonormal:
            proj = w.project_onto(u)
            w = w - proj
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal


class Matrix:
    def __init__(self, rows):
        """
        创建矩阵对象，内部以行列表表示。

        参数:
            rows (iterable of iterables): 矩阵的行集合，每一行为可迭代的数值序列。

        示例:
            >>> M = Matrix([[1,2],[3,4]])
            >>> M.shape
            (2, 2)
        """
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        """
        矩阵与向量或矩阵的乘法（使用 `@` 操作符）。

        - 如果 `other` 是 `Vector`，返回一个 `Vector`，表示矩阵对向量的线性变换。
        - 否则假设 `other` 是 `Matrix` 并返回矩阵乘积。

        注意: 未显式检查维度一致性；若维度不匹配会抛出索引或长度相关错误。

        示例:
            >>> M = Matrix([[1,0],[0,1]])
            >>> v = Vector([2,3])
            >>> M @ v
            Vector([2, 3])
        """
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(sum(
                    self.rows[i][k] * other.rows[k][j]
                    for k in range(self.shape[1])
                ))
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        """
        返回当前矩阵的转置矩阵。

        示例:
            >>> Matrix([[1,2],[3,4]]).transpose()
            Matrix([[1, 3], [2, 4]])
        """
        return Matrix([
            [self.rows[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])

    def rank(self):
        """
        使用行简化的高斯消元法计算矩阵的秩。

        返回:
            int: 矩阵的秩。

        示例:
            >>> Matrix([[1,0],[0,1]]).rank()
            2
        """
        rows = [row[:] for row in self.rows]
        m, n = self.shape
        r = 0
        for col in range(n):
            pivot = None
            for row in range(r, m):
                if abs(rows[row][col]) > 1e-10:
                    pivot = row
                    break
            if pivot is None:
                continue
            rows[r], rows[pivot] = rows[pivot], rows[r]
            scale = rows[r][col]
            rows[r] = [x / scale for x in rows[r]]
            for row in range(m):
                if row != r and abs(rows[row][col]) > 1e-10:
                    factor = rows[row][col]
                    rows[row] = [rows[row][j] - factor * rows[r][j] for j in range(n)]
            r += 1
        return r

    def __repr__(self):
        return f"Matrix({self.rows})"


if __name__ == "__main__":
    print("=== Vectors ===")
    a = Vector([1, 2, 3])
    b = Vector([4, 5, 6])
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a - b = {a - b}")
    print(f"a * 3 = {a * 3}")
    print(f"a · b = {a.dot(b)}")
    print(f"|a| = {a.magnitude():.4f}")
    print(f"â (normalized) = {a.normalize()}")
    print(f"cosine_similarity(a, b) = {a.cosine_similarity(b):.4f}")

    print("\n=== Matrices ===")
    rotation_90 = Matrix([[0, -1], [1, 0]])
    point = Vector([3, 1])
    rotated = rotation_90 @ point
    print(f"Rotate {point} by 90° → {rotated}")

    print("\n=== Angle Between Vectors ===")
    v1 = Vector([1, 0])
    v2 = Vector([0, 1])
    v3 = Vector([1, 1])
    print(f"Angle between {v1} and {v2}: {v1.angle_between(v2):.1f} degrees")
    print(f"Angle between {v1} and {v3}: {v1.angle_between(v3):.1f} degrees")
    print(f"Angle between {v1} and {v1}: {v1.angle_between(v1):.1f} degrees")

    print("\n=== Projection ===")
    a = Vector([3, 4])
    b = Vector([1, 0])
    proj = a.project_onto(b)
    residual = a - proj
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"proj_b(a) = {proj}")
    print(f"residual = {residual}")
    print(f"residual dot b = {residual.dot(b):.6f}")

    print("\n=== Linear Independence ===")
    e1 = Vector([1, 0, 0])
    e2 = Vector([0, 1, 0])
    e3 = Vector([0, 0, 1])
    dep = Vector([2, 1, 0])
    print(f"{{e1, e2, e3}} independent: {is_independent([e1, e2, e3])}")
    print(f"{{e1, e2, 2*e1+e2}} independent: {is_independent([e1, e2, dep])}")

    print("\n=== Gram-Schmidt Orthogonalization ===")
    u1 = Vector([1, 1, 0])
    u2 = Vector([1, 0, 1])
    u3 = Vector([0, 1, 1])
    basis = gram_schmidt([u1, u2, u3])
    for i, vec in enumerate(basis):
        print(f"u{i+1} = {vec}")
    print(f"u1 dot u2 = {basis[0].dot(basis[1]):.6f}")
    print(f"u1 dot u3 = {basis[0].dot(basis[2]):.6f}")
    print(f"u2 dot u3 = {basis[1].dot(basis[2]):.6f}")
    for i, vec in enumerate(basis):
        print(f"|u{i+1}| = {vec.magnitude():.6f}")

    print("\n=== Matrix Rank ===")
    full_rank = Matrix([[1, 0], [0, 1]])
    rank_deficient = Matrix([[1, 2], [2, 4]])
    rectangular = Matrix([[1, 0, 0], [0, 1, 0]])
    print(f"Identity 2x2 rank: {full_rank.rank()}")
    print(f"[[1,2],[2,4]] rank: {rank_deficient.rank()}")
    print(f"[[1,0,0],[0,1,0]] rank: {rectangular.rank()}")

    print("\n=== Neural Network Layer (Matrix x Vector) ===")
    import random
    random.seed(42)
    weights = Matrix([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)])
    input_vec = Vector([1.0, 0.5, -0.3])
    output = weights @ input_vec
    print(f"Input (3D):  {input_vec}")
    print(f"Output (2D): {output}")
    print("^ This is literally what a neural network layer does.")
