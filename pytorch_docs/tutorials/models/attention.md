# 注意力机制

## 数学推导
注意力机制是深度学习中广泛应用的一种技术，可以用于图像分类、自然语言处理等任务中。下面我将为你介绍一下注意力机制的数学推导过程。

假设我们有一个输入向量 $x_i$，和一个包含 $n$ 个元素的向量序列 ${y_1, y_2, \dots, y_n}$，我们需要计算出 $x_i$ 和每个 $y_j$ 的相关性，然后用这些相关性来给每个 $y_j$ 赋予一个权重 $\alpha_j$，最终输出的向量 $r_i$ 是所有 $y_j$ 的加权和。

第一步是计算相关性，我们可以定义一个 $n$ 维向量 $e$，其中每个元素 $e_j$ 表示 $x_i$ 和 $y_j$ 之间的相关性，计算方法如下：

$$e_j = a(x_i, y_j)$$

其中 $a$ 是一个函数，用于计算 $x_i$ 和 $y_j$ 之间的相关性。

第二步是计算权重，我们需要对 $e$ 进行归一化处理，使得所有的权重 $\alpha_j$ 满足 $\sum_{j=1}^{n} \alpha_j = 1$。一种常见的归一化方式是使用 softmax 函数，计算方法如下：

$$\alpha_j = \frac{\exp(e_j)}{\sum_{k=1}^{n} \exp(e_k)}$$

第三步是计算加权和，我们可以将 $y_j$ 和 $\alpha_j$ 相乘，然后将它们的和作为输出 $r_i$：

$$r_i = \sum_{j=1}^{n} \alpha_j y_j$$

注意力机制的数学推导就是以上三个步骤。实际应用中，我们可以将注意力机制嵌入到神经网络中，让模型自动学习如何计算 $a$ 函数，以及如何使用注意力来加强模型的表达能力。