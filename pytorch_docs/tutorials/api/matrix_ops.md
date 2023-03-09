# 向量和矩阵运算

## 常用API
PyTorch 支持多种向量和矩阵运算，以下是其中一些常见的运算：

1. 向量点积

向量点积也称为内积或数量积，是两个向量对应元素乘积的总和，可以使用 torch.dot() 实现。示例如下：

```python
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = torch.dot(a, b)
print(c)  # 输出：tensor(32)
```
2. 矩阵乘法

矩阵乘法是将一个矩阵的每一行与另一个矩阵的每一列对应元素乘积的总和，可以使用 torch.mm() 或 torch.matmul() 实现。示例如下：

```python
import torch

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
c = torch.mm(a, b)
print(c)  # 输出：tensor([[19, 22], [43, 50]])
```

3. 矩阵转置

矩阵转置是将矩阵的行和列互换，可以使用 torch.t() 或者 torch.transpose() 实现。示例如下：

```python
import torch

a = torch.tensor([[1, 2], [3, 4]])
b = torch.t(a)
print(b)  # 输出：tensor([[1, 3], [2, 4]])
```

4. 矩阵求逆

矩阵求逆是指找到一个矩阵，使得该矩阵与原矩阵相乘后得到单位矩阵，可以使用 torch.inverse() 实现。示例如下：

```python
import torch

a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.inverse(a)
print(b)  # 输出：tensor([[-2.,  1.], [ 1.5, -0.5]])
```

5. 矩阵求迹

矩阵求迹是指矩阵主对角线上的元素之和，可以使用 torch.trace() 实现。示例如下：

```python
import torch

a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.trace(a)
print(b)  # 输出：tensor(5.)
```

6. 矩阵行列式

矩阵行列式是一个标量值，它表示一个 n 阶矩阵的所有 n 个行向量组成的空间的有向体积大小。可以使用 torch.det() 实现。示例如下：

```python
import torch

a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.det(a)
print(b)  # 输出：tensor(-2.)
```

7. 矩阵分解

PyTorch 支持多种矩阵分解，包括奇异值分解（SVD）、QR 分解、LU 分解等等。可以使用相应的函数实现。示例如下：

```python
import torch

a = torch.tensor([[1., 2.], [3., 4.]])
# SVD分解
u, s, v = torch.svd(a)
print(u, s, v)
# QR分解
q, r = torch.qr(a)
print(q, r)
# LU分解
l, u = torch.lu(a)
print(l, u)
```

8. 矩阵求解

矩阵求解是指找到一个矩阵 x，使得方程 Ax = b 成立。可以使用 torch.solve() 实现。示例如下：

```python
import torch

a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[-1., -2.], [-3., -4.]])
x, lu = torch.solve(b, a)
print(x)  # 输出：tensor([[ 0., -1.], [ 0., -1.]])
```

9. 矩阵秩
矩阵秩是一个矩阵所拥有的线性无关的行或列的最大数量，可以使用 torch.matrix_rank() 实现。示例如下：

```python
import torch

a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.matrix_rank(a)
print(b)  # 输出：tensor(2)
```

10. 矩阵范数

矩阵范数表示矩阵中元素的绝对值的某种度量，可以使用 torch.norm() 实现。示例如下：

```python
import torch

a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.norm(a)
print(b)  # 输出：tensor(5.4772)
```

11. 矩阵特征值和特征向量

矩阵特征值和特征向量是矩阵在线性代数中的重要概念，可以使用 torch.eig() 实现。示例如下：

```python
import torch

a = torch.tensor([[1., 2.], [3., 4.]])
eigenvalues, eigenvectors = torch.eig(a, eigenvectors=True)
print(eigenvalues)      # 输出：tensor([[-0.3723, 0.0000], [ 5.3723, 0.0000]])
print(eigenvectors)     # 输出：tensor([[-0.8246, -0.4151], [ 0.5658, -0.9094]])
```

12. 矩阵迹

矩阵迹是一个矩阵对角线上元素之和，可以使用 torch.trace() 实现。示例如下：

```python
import torch

a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.trace(a)
print(b)  # 输出：tensor(5.)
```

13. 向量内积

向量内积也称为点积或数量积，可以使用 torch.dot() 实现。示例如下：

```python
import torch

a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])
c = torch.dot(a, b)
print(c)  # 输出：tensor(32.)
```

14. 向量外积

向量外积也称为叉积，可以使用 torch.cross() 实现。示例如下：

```python
import torch

a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])
c = torch.cross(a, b)
print(c)  # 输出：tensor([-3.,  6., -3.])
```

15. 克罗内克积

克罗内克积可以将两个向量组合成一个矩阵，可以使用 torch.kron() 实现。示例如下：

```python
import torch

a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])
c = torch.kron(a, b)
print(c)  # 输出：tensor([ 4.,  5.,  6.,  8., 10., 12., 12., 15., 18.])
```

## BLAS 和 LAPACK 运算
BLAS（Basic Linear Algebra Subprograms）和 LAPACK（Linear Algebra Package）是数值线性代数库，提供了一些高效的基本线性代数运算，如矩阵乘法、矩阵求逆、矩阵分解等。PyTorch 中也内置了一些 BLAS 和 LAPACK 的函数，可以方便地进行这些运算。

下面列举一些 PyTorch 中常用的 BLAS 和 LAPACK 运算的 API：

1. 矩阵乘法
- torch.matmul()：矩阵乘法函数。
- @：矩阵乘法运算符。

2. 矩阵求逆
- torch.inverse()：矩阵求逆函数。

3. 矩阵分解
- torch.svd()：SVD 分解函数。
- torch.lu()：LU 分解函数。

4. 其他常用的 BLAS 和 LAPACK 运算
- torch.addmm()：计算矩阵-矩阵乘积并加上一个矩阵。
- torch.addmv()：计算矩阵-向量乘积并加上一个向量。
- torch.addr()：计算向量的外积并加上一个矩阵。
- torch.bmm()：计算批次的矩阵-矩阵乘积。
- torch.cross()：计算两个向量的叉积。
- torch.det()：计算矩阵的行列式。
- torch.eig()：计算实对称矩阵的特征值和特征向量。
- torch.ger()：计算两个向量的外积。
- torch.inverse()：计算矩阵的逆。
- torch.lstsq()：计算最小二乘解。
- torch.mm()：计算矩阵-矩阵乘积。
- torch.mv()：计算矩阵-向量乘积。
- torch.norm()：计算矩阵或向量的范数。
- torch.pinverse()：计算矩阵的摄动逆。
- torch.qr()：计算矩阵的 QR 分解。
- torch.solve()：求解线性方程组。
- torch.svd()：计算矩阵的 SVD 分解。


5. BLAS 和 LAPACK 的数据类型
PyTorch 中支持的 BLAS 和 LAPACK 运算支持多种数据类型，包括：

- float32（单精度浮点数）
- float64（双精度浮点数）
- complex64（单精度复数）
- complex128（双精度复数）

对于某些 BLAS 和 LAPACK 运算，还支持一些特殊的数据类型，例如 half（半精度浮点数）和 bfloat16（Bfloat16 浮点数）等。

6. PyTorch 中的 BLAS 和 LAPACK 实现
PyTorch 中的 BLAS 和 LAPACK 运算可以使用多种实现方式，包括：

- CPU 实现：使用 OpenBLAS 或者 MKL 等 BLAS 和 LAPACK 库。
- GPU 实现：使用 cuBLAS 或者 MAGMA 等 BLAS 和 LAPACK 库。
默认情况下，PyTorch 会使用 MKL 库进行 CPU 实现，使用 cuBLAS 库进行 GPU 实现。如果需要使用其他的 BLAS 或者 LAPACK 库，可以手动安装并配置 PyTorch 的环境变量。

7. BLAS 和 LAPACK 运算的优化
为了提高 BLAS 和 LAPACK 运算的性能，PyTorch 中采用了多种优化技术，包括：

- 张量转置和内存对齐：将张量按照一定的规则进行转置和内存对齐，可以提高 BLAS 和 LAPACK 运算的效率。
- 并行计算：利用多线程和多进程技术，将 BLAS 和 LAPACK 运算进行并行计算，提高计算效率。
- 混合精度计算：使用半精度浮点数等低精度的数据类型进行计算，可以加速 BLAS 和 LAPACK 运算的速度。

总之，PyTorch 提供了丰富的 BLAS 和 LAPACK 运算 API，支持多种数据类型和实现方式，并且采用了多种优化技术，可以满足不同场景下的运算需求，并提高运算效率。





