import optuna
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn.functional as F

# 数据预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# 加载数据
train_dataset = datasets.MNIST(root='./data/mnist', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data/mnist', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义设备（GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型
class Net(torch.nn.Module):
    def __init__(self, conv1_out_channels, conv2_out_channels, fc_units):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, conv1_out_channels, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(conv2_out_channels * 4 * 4, fc_units),
            torch.nn.ReLU(),
            torch.nn.Linear(fc_units, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(batch_size, -1)  # flatten
        x = self.fc(x)
        return x

# 超参数调优目标函数
def objective(trial):
    # 定义超参数搜索空间
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.5, 0.9)
    conv1_out_channels = trial.suggest_int("conv1_out_channels", 8, 64, step=8)
    conv2_out_channels = trial.suggest_int("conv2_out_channels", 8, 64, step=8)
    fc_units = trial.suggest_int("fc_units", 32, 128, step=16)

    # 数据加载器（动态调整批大小）
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 模型定义并移至GPU
    model = Net(conv1_out_channels, conv2_out_channels, fc_units).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # 训练与测试
    def train(epoch):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, target) in enumerate(train_loader):
            inputs, target = inputs.to(device), target.to(device)  # 数据移至GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # 每隔一定的批次输出损失
            if batch_idx % 100 == 0:  # 每100个批次输出一次
                print(f"Train Epoch {epoch+1} [{batch_idx * len(inputs)}/{len(train_loader.dataset)}] "
                      f"Loss: {loss.item():.6f}")

        # 输出一个epoch的平均损失
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")

    def test():
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # 数据移至GPU
                outputs = model(inputs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return accuracy

    # 训练和验证
    EPOCH = 5  # 每次试验训练 5 个 epoch
    for epoch in range(EPOCH):
        train(epoch)
    acc = test()
    return acc  # 返回测试集准确率

# 使用 Optuna 优化
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    # 打印最佳参数
    print("Best hyperparameters:", study.best_params)

    # 使用最佳参数重新训练模型
    best_params = study.best_params
    final_model = Net(
        conv1_out_channels=best_params["conv1_out_channels"],
        conv2_out_channels=best_params["conv2_out_channels"],
        fc_units=best_params["fc_units"],
    ).to(device)  # 模型移至GPU

    final_optimizer = torch.optim.SGD(
        final_model.parameters(), lr=best_params["lr"], momentum=best_params["momentum"]
    )
    final_train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)

    # 训练和测试最终模型
    EPOCH = 10
    acc_list = []
    for epoch in range(EPOCH):
        final_model.train()
        running_loss = 0.0
        for inputs, target in final_train_loader:
            inputs, target = inputs.to(device), target.to(device)  # 数据移至GPU
            final_optimizer.zero_grad()
            outputs = final_model(inputs)
            loss = torch.nn.CrossEntropyLoss()(outputs, target)
            loss.backward()
            final_optimizer.step()
            running_loss += loss.item()

        # 输出一个epoch的平均损失
        avg_loss = running_loss / len(final_train_loader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.6f}")

        # 输出测试集准确率
        final_model.eval()
        acc = sum(
            (torch.argmax(final_model(inputs.to(device)), dim=1) == labels.to(device)).sum().item()
            for inputs, labels in test_loader
        ) / len(test_dataset)
        acc_list.append(acc)
        print(f"Epoch {epoch + 1}: Test Accuracy = {acc * 100:.2f}%")

    # 绘制最终模型准确率曲线
    plt.plot(acc_list)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Final Model Test Accuracy")
    plt.show()
