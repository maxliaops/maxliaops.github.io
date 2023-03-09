# 张量

## 张量 API
PyTorch 中最基本的数据结构是张量（tensor），张量可以理解为一个多维数组。PyTorch 提供了许多张量的操作函数，下面列举一些常用的张量 API：

1. 创建张量
```python
import torch

# 创建一个 3x3 的张量
a = torch.Tensor(3, 3)

# 创建一个 2x2 的随机张量，元素取值范围为 [0, 1)
b = torch.rand(2, 2)

# 创建一个 1x2 的张量，元素取值范围为 [0, 10)
c = torch.randint(low=0, high=10, size=(1, 2))

# 创建一个 2x2 的全零张量
d = torch.zeros(2, 2)

# 创建一个 2x2 的全一张量
e = torch.ones(2, 2)
```

2. 张量的基本操作
```python
import torch

# 张量加法
a = torch.rand(2, 2)
b = torch.rand(2, 2)
c = a + b

# 张量乘法
a = torch.rand(2, 2)
b = torch.rand(2, 2)
c = torch.mm(a, b)

# 张量转置
a = torch.rand(2, 3)
b = a.t()

# 张量求和
a = torch.rand(2, 2)
b = torch.sum(a)

# 张量平均值
a = torch.rand(2, 2)
b = torch.mean(a)

# 张量标准差
a = torch.rand(2, 2)
b = torch.std(a)
```

3. 张量的索引和切片
```python
import torch

# 张量索引
a = torch.rand(3, 3)
b = a[0, 1]

# 张量切片
a = torch.rand(3, 3)
b = a[:, 1]  # 取第二列
c = a[1:, :]  # 取第二行及以后的所有行
d = a[1:, 1:]  # 取第二行及以后的所有行和第二列及以后的所有列
```

4. 张量形状变换
```python
import torch

# 张量转换形状
a = torch.rand(2, 2)
b = a.view(4)  # 转换为一维张量
c = a.view(1, 4)  # 转换为一行四列的二维张量
d = a.view(-1, 1)  # 转换为 n 行一列的二维张量，其中 n 由张量大小自动推断

# 张量扩展形状
a = torch.rand(2, 2)
b = a.unsqueeze(0)  # 在第一维增加一维
c = a.unsqueeze(1)  # 在第二维增加一维
d = a.unsqueeze(2)  # 在第三维增加一维

# 张量缩减形状
a = torch.rand(2, 2, 2)
b = a.squeeze()  # 缩减所有大小为1的维度
c = a.squeeze(0)  # 缩减第一维大小为1的维度
d = a.squeeze(-1)  # 缩减最后一维大小为1的维度
```

5. 张量的拼接和拆分
```python
import torch

# 张量拼接
a = torch.rand(2, 2)
b = torch.rand(2, 2)
c = torch.cat((a, b), dim=0)  # 在第一维拼接
d = torch.cat((a, b), dim=1)  # 在第二维拼接

# 张量拆分
a = torch.rand(2, 4)
b, c = torch.split(a, split_size_or_sections=2, dim=1)  # 在第二维拆分成大小为2的两个张量
d, e, f = torch.chunk(a, chunks=3, dim=1)  # 在第二维均分成大小为1的三个张量
```
