import minitorch
import numpy as np

# 参数初始化函数
def RParam(*shape):
    return minitorch.Parameter(2 * (np.random.rand(*shape) - 0.5))

# 日志函数
def default_log_fn(epoch, total_loss, correct, losses):
    print(f"Epoch {epoch}, Loss: {total_loss:.4f}, Correct: {correct}")

class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = self.add_parameter("weights", RParam(in_size, out_size))
        self.bias = self.add_parameter("bias", RParam(out_size))

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.view(1, -1)
        return x @ self.weights.value + self.bias.value

class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        x = self.layer1(x).relu()
        x = self.layer2(x).relu()
        x = self.layer3(x).sigmoid()
        return x

class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        # 打印输入 x 的类型和值
        print(f"x: {x}, type: {type(x)}")
        
        # 确保 x 是一个长度为 2 的序列
        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) != 2:
                raise ValueError(f"Input x must have exactly 2 features, but got {len(x)}.")
        else:
            raise TypeError("Input x must be a sequence (list, tuple, or ndarray) of length 2.")
        
        # 将 x 转换为张量
        x_tensor = minitorch.tensor(x)
        print(f"x_tensor before view: shape {x_tensor.shape}, size {x_tensor.size}")
        
        # 调整形状为 (1, -1)
        x_tensor = x_tensor.view(1, -1)
        print(f"x_tensor after view: shape {x_tensor.shape}, size {x_tensor.size}")
        
        return self.model.forward(x_tensor)


        
       
    def run_many(self, X):
        X_tensor = minitorch.tensor(X)
        return self.model.forward(X_tensor)

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        
        print(f"Data X shape: {data.X.shape}, Data y shape: {data.y.shape}")
        print(f"First data point X[0]: {data.X[0]}")
    
    # 其余代码保持不变

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # 前向传播
            out = self.model.forward(X).view(-1)
            # 计算损失
            loss = -(y * out.log() + (1 - y) * (1 - out).log()).mean()
            total_loss = loss.item()

            # 反向传播
            loss.backward()
            # 更新参数
            optim.step()

            # 计算准确率
            predictions = (out.detach() > 0.5).float()
            correct = (predictions == y).sum().item()

            # 日志记录
            if epoch % 10 == 0 or epoch == self.max_epochs:
                log_fn(epoch, total_loss, correct, [])

if __name__ == "__main__":
    PTS = 50  # 数据点数量
    HIDDEN = 10  # 隐藏层大小
    RATE = 0.1  # 学习率
    data = minitorch.datasets["Simple"](PTS)
    trainer = TensorTrain(HIDDEN)
    trainer.train(data, RATE)
