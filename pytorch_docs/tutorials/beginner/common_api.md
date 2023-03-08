
# PyTorch常用API
PyTorch是一个非常强大的深度学习框架，提供了很多常用的API。以下是一些常用的API：

## 张量操作
1. 创建张量
```python
import torch

# 创建一个形状为(2, 3)的浮点型张量，初始值为0
x = torch.zeros(2, 3)

# 创建一个形状为(2, 3)的浮点型张量，初始值为1
y = torch.ones(2, 3)

# 创建一个形状为(2, 3)的浮点型张量，随机初始化
z = torch.randn(2, 3)
```
2. 张量加法
```python
# 张量加法
result = x + y
```
3. 张量乘法
```python
# 张量乘法
result = torch.mm(x, y.t()) # 矩阵乘法，需要转置
```
4. 张量索引和切片
```python
# 张量索引和切片
z = torch.randn(3, 4, 5)
print(z[0]) # 取第0个维度的张量
print(z[:, 0, :]) # 取第1个维度的第0个元素，第3个维度的所有元素
```
## 模型构建与训练
1. 定义模型
```python
import torch.nn as nn

class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(16 * 13 * 13, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16 * 13 * 13)
        x = self.fc1(x)
        return x

model = YourModel()
```
2. 定义损失函数和优化器
```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
```
3. 模型训练
```python
# 模型训练
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print('Epoch %d, loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))
```
## 数据处理
1. 数据加载
```python
from torch.utils.data import DataLoader, Dataset

class YourDataset(Dataset):
    def __init__(self, ...):
        ...

    def __getitem__(self, index):
        ...

    def __len__(self):
        ...

train_dataset = YourDataset(...)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
```
2. 数据增强
```python
from torchvision.transforms import transforms

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
```
3. 数据预处理
```python
from PIL import Image

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = transform_train(img)
    img = img.unsqueeze(0) # 将张量变为4D张量，第0个维度表示batch_size
    return img
```
## 其他常用API
1. GPU加速
```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs, labels = data[0].to(device), data[1].to(device)
```
2. 模型保存与加载
```python
# 保存模型
torch.save(model.state_dict(), 'model.pth')

# 加载模型
model.load_state_dict(torch.load('model.pth'))
```
3. 学习率调度器
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(num_epochs):
    ...
    scheduler.step()
```
4. 计算指标
```python
from sklearn.metrics import accuracy_score

def compute_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total
```

## PyTorch张量操作API
PyTorch 张量操作 API 提供了许多用于操作张量（多维数组）的函数和方法。下面是一些常见的张量操作：

1. 创建张量：

- `torch.tensor(data, dtype=None, device=None)`：从数据中创建张量。

- `torch.zeros(size, dtype=None, device=None)`：创建全零张量。

- `torch.ones(size, dtype=None, device=None)`：创建全一张量。

- `torch.eye(n, m=None, dtype=None, device=None)`：创建单位矩阵。

2. 张量形状操作：

- `torch.view(size)`：改变张量形状。

- `torch.squeeze(input, dim=None)`：去除维度为 1 的轴。

- `torch.unsqueeze(input, dim)`：增加维度为 1 的轴。

- `torch.reshape(input, shape)`：改变张量形状。

3. 张量数学操作：

- `torch.add(x, y)`：对张量 x 和 y 进行加法操作。

- `torch.sub(x, y)`：对张量 x 和 y 进行减法操作。

- `torch.mul(x, y)`：对张量 x 和 y 进行乘法操作。

- `torch.div(x, y)`：对张量 x 和 y 进行除法操作。

4. 张量统计操作：

- `torch.mean(input)`：计算张量的均值。

- `torch.std(input)`：计算张量的标准差。

- `torch.max(input)`：计算张量的最大值。

- `torch.min(input)`：计算张量的最小值。

5. 张量逻辑操作：

- `torch.eq(x, y)`：对张量 x 和 y 进行逐元素相等操作。

- `torch.gt(x, y)`：对张量 x 和 y 进行逐元素大于操作。

- `torch.lt(x, y)`：对张量 x 和 y 进行逐元素小于操作。

- `torch.ge(x, y)`：对张量 x 和 y 进行逐元素大于等于操作。

- `torch.le(x, y)`：对张量 x 和 y 进行逐元素小于等于操作。

6. 其他常见操作：

- `torch.cat(tensors, dim=0)`：将多个张量在指定轴上拼接起来。

- `torch.stack(tensors, dim=0)`：将多个张量沿着新轴进行堆叠。

- `torch.transpose(input, dim0, dim1)`：交换张量的两个维度。

- `torch.split(tensor, split_size_or_sections, dim=0)`：在指定维度上将张量分成多个子张量。

- `torch.chunk(input, chunks, dim=0)`：在指定维度上将张量分成多个块。

## PyTorch神经网络API
PyTorch 神经网络 API 提供了一组工具和函数，以构建和训练神经网络。下面是一些常见的神经网络 API：

1. nn.Module：所有神经网络层都从此类继承。它提供了许多函数，例如 forward()，可以自定义神经网络的前向传递过程。

2. nn.Sequential：这个类用于将多个层组合成一个序列。

3. nn.Conv2d：用于卷积层，可以对图像进行卷积操作。

4. nn.MaxPool2d：用于池化层，可以对图像进行最大池化操作。

5. nn.Linear：用于全连接层，可以将输入张量映射到输出张量。

6.  nn.Dropout：用于正则化层，可以在训练过程中随机丢弃一些神经元。

7. nn.BatchNorm2d：用于批归一化层，可以对输入数据进行标准化处理。

8. nn.CrossEntropyLoss：用于计算分类问题的损失函数。

9. optim：PyTorch 中的优化器模块，可以使用 SGD、Adam、RMSprop 等优化算法来更新神经网络的参数。

## PyTorch数据处理API
PyTorch 提供了许多数据处理 API，可以用于加载、转换和操作数据集。以下是一些常用的 PyTorch 数据处理 API：

1. `torch.utils.data.Dataset`：用于表示数据集的抽象类。我们可以通过继承该类来创建自己的数据集。必须实现 __getitem__() 和 __len__() 方法。

2. `torch.utils.data.DataLoader`：用于加载数据集的类。可以设置批次大小、是否打乱数据和多线程等参数。

3. `torchvision.datasets`：提供了许多常见的计算机视觉数据集，例如 MNIST、CIFAR10 等。

4. `torchvision.transforms`：提供了许多数据转换函数，例如对图像进行缩放、裁剪、翻转和归一化等。



