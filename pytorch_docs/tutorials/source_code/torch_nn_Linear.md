# nn.Linear

## 源代码实现
以下是 PyTorch 中 nn.Linear 类的源代码实现：

```python
import math
from typing import Any

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from .. import functional as F
from .. import init
from .module import Module

class Linear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 进行权重初始化
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        # 线性变换 y = x * W^T + b
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)
```

nn.Linear 类继承自 Module 类，是一个 PyTorch 中的神经网络层，实现了线性变换 $y = xW^T+b$。在初始化时，需要传入输入和输出的特征维度，并且可以选择是否有偏置。在前向传播时，调用 F.linear 函数实现线性变换。其中，self.weight 和 self.bias 都是 Parameter 类型的张量，会被自动加入到参数列表中，并在反向传播时被更新。在重置参数时，nn.Linear 使用了 PyTorch 中的 init 模块进行权重初始化。extra_repr 方法用于输出模型的信息。

## 用法
nn.Linear 是 PyTorch 中的一个类，用于创建一个线性变换的层（也称为全连接层）。它的构造函数有两个参数：输入特征数和输出特征数。下面是一个简单的示例，演示如何使用 nn.Linear 类来创建一个线性层：

```python
import torch.nn as nn

# 创建一个线性层，输入特征数为10，输出特征数为5
linear_layer = nn.Linear(10, 5)

# 创建一个输入张量，大小为 [batch_size, 10]
input_tensor = torch.randn(32, 10)

# 将输入张量传递给线性层进行线性变换，输出张量大小为 [batch_size, 5]
output_tensor = linear_layer(input_tensor)

# 输出张量的大小为 [32, 5]
print(output_tensor.size())
```
在上面的示例中，我们首先使用 nn.Linear 创建了一个线性层，输入特征数为 10，输出特征数为 5。然后我们创建了一个输入张量，大小为 [batch_size, 10]，其中 batch_size 表示批量大小。接着，我们将输入张量传递给线性层进行线性变换，并将输出张量保存在 output_tensor 中。最后，我们输出了 output_tensor 的大小，结果为 [32, 5]。