# 降维操作

## 常用API
PyTorch 中的 Reduction 操作可以将一个多维张量沿着指定的维度进行降维操作，得到一个标量或一个降维后的张量。

常见的 Reduction 操作包括：

1. 求和操作
```python
torch.sum(input, dim=None, keepdim=False, out=None)  # 沿着指定的维度求和
```

2. 求平均值操作
```python
torch.mean(input, dim=None, keepdim=False, out=None)  # 沿着指定的维度求平均值
```

3. 求最大值、最小值、均值和方差操作
```python
torch.max(input, dim=None, keepdim=False, out=None)  # 沿着指定的维度求最大值
torch.min(input, dim=None, keepdim=False, out=None)  # 沿着指定的维度求最小值
torch.mean(input, dim=None, keepdim=False, out=None)  # 沿着指定的维度求平均值
torch.var(input, dim=None, unbiased=True, keepdim=False, out=None)  # 沿着指定的维度求方差
```

4. 计数操作
```python

torch.numel(input)  # 计算张量的元素总数
```

5. 其他操作
```python
torch.prod(input, dim=None, keepdim=False, out=None)  # 沿着指定的维度求积
torch.std(input, dim=None, unbiased=True, keepdim=False, out=None)  # 沿着指定的维度求标准差
```

这些 Reduction 操作都有一些共同的参数，其中 dim 指定要降维的维度，keepdim 指定是否保持原始张量的维度大小，out 指定输出张量的形状。在参数列表中未提及的参数通常为可选参数，有默认值。