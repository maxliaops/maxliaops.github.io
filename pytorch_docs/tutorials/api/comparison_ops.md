# 比较操作

## 常用API
PyTorch 中的 Comparison 操作用于比较两个张量或标量的大小关系，并返回一个布尔类型的张量。常见的 Comparison 操作包括：

`. 比较运算符
```python
torch.eq(input, other, out=None)  # 判断张量 input 和 other 是否相等
torch.ge(input, other, out=None)  # 判断张量 input 是否大于等于 other
torch.gt(input, other, out=None)  # 判断张量 input 是否大于 other
torch.le(input, other, out=None)  # 判断张量 input 是否小于等于 other
torch.lt(input, other, out=None)  # 判断张量 input 是否小于 other
torch.ne(input, other, out=None)  # 判断张量 input 和 other 是否不相等
```

1. 其他运算符
```python
torch.equal(input, other)  # 判断张量 input 和 other 是否完全相等
torch.isinf(tensor)  # 判断张量 tensor 中的元素是否为无穷大
torch.isnan(tensor)  # 判断张量 tensor 中的元素是否为 NaN
```

这些 Comparison 操作都有一些共同的参数，其中 input 表示输入张量，other 表示与输入张量进行比较的张量或标量，out 表示输出张量的形状。在参数列表中未提及的参数通常为可选参数，有默认值。