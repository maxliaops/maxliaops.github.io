- [循环神经网络(RNN)](#循环神经网络rnn)
  - [原理](#原理)
  - [数学推导](#数学推导)
  - [使用numpy从零开始实现循环神经网络](#使用numpy从零开始实现循环神经网络)


# 循环神经网络(RNN)

## 原理
循环神经网络（Recurrent Neural Network，RNN）是一类用于处理序列数据的神经网络。与前馈神经网络（Feedforward Neural Network）不同，RNN 可以使用其内部的状态（hidden state）来存储序列数据的信息，并且可以在每一个时间步（time step）使用相同的权重来处理不同的输入。

循环神经网络的基本原理是使用一个循环的结构来处理序列数据。具体来说，每个时间步的输入 $x_t$ 与上一个时间步的状态 $h_{t-1}$ 一起经过一系列的计算得到当前时间步的输出 $y_t$ 和当前时间步的状态 $h_t$。因此，循环神经网络可以看做是一个动态的神经网络，它能够对不同长度的序列数据进行处理，并且能够在处理序列数据的同时维持数据的顺序和时序关系。

RNN 的计算公式如下：

$$h_t = f(Ux_t + Wh_{t-1} + b)$$

其中，$U$、$W$、$b$ 分别为权重矩阵和偏置向量，$f$ 是激活函数，$h_t$ 是当前时间步的状态，$x_t$ 是当前时间步的输入。

在上式中，$Ux_t$ 是当前时间步的输入与其对应的权重矩阵相乘的结果，$Wh_{t-1}$ 是上一个时间步的状态与其对应的权重矩阵相乘的结果。这两个结果加上偏置向量 $b$ 后，经过激活函数 $f$ 的作用，就得到了当前时间步的状态 $h_t$。

可以看出，循环神经网络的计算过程包含了一个循环，每个时间步的状态都依赖于上一个时间步的状态和当前时间步的输入。在训练过程中，循环神经网络的参数（权重和偏置）通过反向传播算法进行更新，以使网络能够适应不同的序列数据，并且能够对序列数据进行预测和生成。

## 数学推导

循环神经网络（Recurrent Neural Network，RNN）可以通过数学推导来更深入地理解其原理和计算过程。

假设我们有一个时间序列 $(x_1, x_2, ..., x_T)$，其中 $x_t \in \mathbb{R}^{d}$ 表示时间步 $t$ 的输入向量，$d$ 表示输入向量的维度。循环神经网络的计算过程可以表示为以下递归式：

$$h_t = f(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

$$y_t = W_{hy} h_t + b_y$$

其中 $h_t$ 表示时间步 $t$ 的隐藏状态（hidden state），$y_t$ 表示时间步 $t$ 的输出，$W_{xh}$、$W_{hh}$、$W_{hy}$ 分别表示输入到隐藏状态、隐藏状态到隐藏状态、隐藏状态到输出的权重矩阵，$b_h$、$b_y$ 分别表示隐藏状态和输出的偏置向量，$f$ 表示激活函数。

通过上式，我们可以得到一个循环神经网络的计算过程。具体地，当时间步 $t=1$ 时，由于 $h_0$ 的值未知，我们可以将其设置为全零向量，即 $h_0 = 0$。然后，我们可以使用递归式计算出 $h_t$ 和 $y_t$，并将 $h_t$ 作为下一个时间步的隐藏状态输入。

在训练过程中，我们需要定义损失函数并通过反向传播算法来更新网络的参数。具体地，假设我们有一个标签序列 $(y_1, y_2, ..., y_T)$，其中 $y_t \in \mathbb{R}^{q}$ 表示时间步 $t$ 的标签向量，$q$ 表示标签向量的维度。我们可以使用交叉熵损失函数来度量网络的预测值和真实值之间的差异：

$$\mathcal{L} = - \sum_{t=1}^{T} \sum_{i=1}^{q} y_{t,i} \log \hat{y}_{t,i}$$

其中 $\hat{y}_{t,i}$ 表示网络在时间步 $t$ 预测的标签向量中的第 $i$ 个分量的值。

通过反向传播算法，我们可以计算出损失函数对网络参数的梯度，并使用梯度下降法来更新网络参数，从而使网络的预测结果更加准确。

需要注意的是，在训练循环神经网络时，由于计算过程包含了递归式的循环，因此需要使用反向传播算法的链式法则来计算梯度。具体地，我们需要将梯度在时间轴上展开，然后计算每个时间步的梯度。

在循环神经网络中，每个时间步的损失函数对网络参数的梯度可以通过时间反向传播（Backpropagation Through Time，BPTT）算法来计算。BPTT 算法可以被视为标准的反向传播算法在时间轴上的扩展，即将前向传播和反向传播算法推广到时间序列数据上。

具体地，BPTT 算法通过将时间轴上的网络展开成一个前向传播的计算图和一个反向传播的计算图来计算梯度。在前向传播过程中，我们按时间顺序计算出网络的输出和隐藏状态，并将其存储在一个矩阵中。在反向传播过程中，我们按时间反向传播误差信号，并计算出每个时间步的梯度。然后，我们可以使用梯度下降算法来更新网络参数。

需要注意的是，在计算梯度时，由于时间轴上的展开，导致每个时间步的梯度都会对前面所有时间步的梯度产生影响，因此在实际训练中，BPTT 算法往往会出现梯度消失或梯度爆炸的问题。为了解决这个问题，通常会使用一些技术，如梯度剪裁（gradient clipping）或 LSTM（Long Short-Term Memory）网络。

总之，循环神经网络通过在时间序列数据上的递归计算和时间反向传播算法，可以有效地处理时序数据，并在语音识别、自然语言处理、股票预测等领域取得了广泛的应用。

在训练过程中，循环神经网络(RNN)需要存储所有时间步的隐藏状态，因为后面的隐藏状态依赖于前面的隐藏状态。这是为了计算梯度并更新模型的权重和偏置。

在每个时间步，RNN接收输入和前一个时间步的隐藏状态，计算当前时间步的隐藏状态，并将其传递给下一个时间步。因此，每个时间步的隐藏状态都需要被存储，以便在计算梯度和更新权重时使用。

但是，在推理过程中，RNN只需要存储当前时间步的隐藏状态，因为它只需要根据当前输入计算输出，并将当前隐藏状态传递给下一个时间步。因此，推理时只需要存储当前时间步的隐藏状态，而不需要存储之前的隐藏状态。

总之，在训练过程中，RNN需要存储所有时间步的隐藏状态，而在推理过程中，只需要存储当前时间步的隐藏状态。

## 使用numpy从零开始实现循环神经网络
```python
class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化参数
        self.Wxh = np.random.randn(input_size, hidden_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(hidden_size, output_size) * 0.01
        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, output_size))
        
        # 保存中间状态
        self.h = {}
        self.y = {}
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def forward(self, x):
        # 初始化隐藏状态
        self.h[-1] = np.zeros((1, self.Whh.shape[0]))
        
        # 前向传播
        for t in range(x.shape[0]):
            self.h[t] = self.sigmoid(np.dot(x[t], self.Wxh) + np.dot(self.h[t-1], self.Whh) + self.bh)
            self.y[t] = np.dot(self.h[t], self.Why) + self.by
        
        # 返回最后一个时间步的输出
        return self.y[x.shape[0]-1]
    
    def backward(self, x, y_true, learning_rate=0.1):
        # 初始化梯度
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros((1, self.Whh.shape[0]))
        
        # 反向传播
        for t in reversed(range(x.shape[0])):
            dy = self.y[t] - y_true[t]
            dWhy += np.dot(self.h[t].T, dy)
            dby += dy
            dh = np.dot(dy, self.Why.T) + dh_next
            dh_raw = self.h[t] * (1 - self.h[t]) * dh
            dbh += dh_raw
            dWxh += np.dot(x[t].reshape(1,-1).T, dh_raw)
            dWhh += np.dot(self.h[t-1].T, dh_raw)
            dh_next = np.dot(dh_raw, self.Whh.T)
        
        # 更新参数
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby

```