# 逐元素操作

## 常用API
PyTorch 中有许多逐元素（Pointwise）操作的 API，常见的包括：

1. 加法、减法、乘法、除法
```python
torch.add(input, other, *, alpha=1, out=None)  # 加法
torch.sub(input, other, *, out=None)  # 减法
torch.mul(input, other, out=None)  # 乘法
torch.div(input, other, out=None)  # 除法
```

2. 指数函数、对数函数、幂函数、开方函数
```python
torch.exp(input, out=None)  # 指数函数
torch.log(input, out=None)  # 自然对数函数
torch.log10(input, out=None)  # 以 10 为底的对数函数
torch.log2(input, out=None)  # 以 2 为底的对数函数
torch.pow(input, exponent, out=None)  # 幂函数
torch.sqrt(input, out=None)  # 开方函数
```

3. 三角函数、反三角函数、双曲函数
```python
torch.sin(input, out=None)  # 正弦函数
torch.cos(input, out=None)  # 余弦函数
torch.tan(input, out=None)  # 正切函数
torch.asin(input, out=None)  # 反正弦函数
torch.acos(input, out=None)  # 反余弦函数
torch.atan(input, out=None)  # 反正切函数
torch.sinh(input, out=None)  # 双曲正弦函数
torch.cosh(input, out=None)  # 双曲余弦函数
torch.tanh(input, out=None)  # 双曲正切函数
```

4. 比较运算
```python
torch.eq(input, other, out=None)  # 逐元素比较是否相等
torch.ne(input, other, out=None)  # 逐元素比较是否不相等
torch.ge(input, other, out=None)  # 逐元素比较是否大于等于
torch.le(input, other, out=None)  # 逐元素比较是否小于等于
torch.gt(input, other, out=None)  # 逐元素比较是否大于
torch.lt(input, other, out=None)  # 逐元素比较是否小于
```

5. 其他逐元素操作
```python
torch.abs(input, out=None)  # 绝对值
torch.acos(input, out=None)  # 反余弦函数
torch.ceil(input, out=None)  # 向上取整
torch.clamp(input, min=None, max=None, out=None)  # 限制在[min,max]区间内
torch.floor(input, out=None)  # 向下取整
torch.fmod(input, other, out=None)  # 取模
torch.reciprocal(input, out=None)  # 倒数
torch.neg(input, out=None)  # 取相反数
torch.round(input, out=None)  # 四舍五入
torch.sign(input, out=None)  # 符号函数
torch.sigmoid(input, out=None)  # sigmoid
```

## Pointwise 运算与Element-wise运算区别与联系
在 PyTorch 中，Pointwise 运算和 Element-wise 运算都是对输入张量中的每个元素进行不同的数学运算。它们的主要区别在于运算的形式和目的。

Pointwise 运算通常指逐点运算，即按照相同位置上的元素进行操作，常见的 Pointwise 运算包括：

- 加法：torch.add()；
- 减法：torch.sub()；
- 乘法：torch.mul()；
- 除法：torch.div()。
这些运算通常用于执行像素级别的图像处理任务，例如图像的亮度、对比度调整等。

Element-wise 运算通常指对输入张量中的每个元素进行不同的数学运算，常见的 Element-wise 运算包括：

- 指数运算：torch.exp()；
- 对数运算：torch.log()；
- 幂运算：torch.pow()；
- 三角函数运算：torch.sin()；
- 双曲线函数运算：torch.sinh()。
这些运算通常用于执行数学运算和科学计算任务。

虽然 Pointwise 运算和 Element-wise 运算在形式上存在一定的差异，但它们都是逐个元素进行操作的，并且都可用于处理张量数据。此外，许多常见的数学运算都既可以通过 Pointwise 运算实现，也可以通过 Element-wise 运算实现，具体使用哪种方法取决于应用场景和计算要求。