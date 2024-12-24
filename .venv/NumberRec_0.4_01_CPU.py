import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import optuna  # 导入Optuna库，进行自动超参数调优

# 超参数设置 ------------------------------------------------------------------------------------
batch_size = 64  # 设置每个批次的大小为64
EPOCH = 10  # 设置训练的总轮数为10

# 数据集准备 ------------------------------------------------------------------------------------
# 对数据进行转换，转为tensor并进行标准化
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# 加载训练和测试数据集
train_dataset = datasets.MNIST(root='./data/mnist', train=True, transform=transform)
test_dataset = datasets.MNIST(root='./data/mnist', train=False, transform=transform)

# 使用DataLoader将数据加载到内存中，并设置batch_size和是否打乱数据
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 可视化部分，显示前12张训练图片及其标签
fig = plt.figure()
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.tight_layout()
    plt.imshow(train_dataset.train_data[i], cmap='gray', interpolation='none')
    plt.title("Labels: {}".format(train_dataset.train_labels[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

# 定义神经网络模型 -----------------------------------------------------------------------------------------
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 第一层卷积层：输入1个通道，输出10个通道，卷积核大小为5
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),  # 激活函数ReLU
            torch.nn.MaxPool2d(kernel_size=2),  # 池化层，kernel_size=2
        )
        # 第二层卷积层：输入10个通道，输出20个通道，卷积核大小为5
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),  # 激活函数ReLU
            torch.nn.MaxPool2d(kernel_size=2),  # 池化层，kernel_size=2
        )
        # 全连接层：输入320个特征，输出50个特征，然后输出10个特征（对应数字0-9）
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)  # 获取输入的batch_size
        x = self.conv1(x)  # 第一层卷积
        x = self.conv2(x)  # 第二层卷积
        x = x.view(batch_size, -1)  # 将卷积输出展平，变成全连接层的输入
        x = self.fc(x)  # 经过全连接层
        return x  # 输出一个10维向量，对应每个类别的得分

# 定义损失函数和优化器 ----------------------------------------------------------------------
# 使用交叉熵损失函数（CrossEntropyLoss）进行分类任务
criterion = torch.nn.CrossEntropyLoss()

# 定义一个函数来创建优化器，使用SGD优化器，并传入学习率和动量
def create_optimizer(model, learning_rate, momentum):
    return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# 训练和测试函数 --------------------------------------------------------------------------------------

# 训练函数，每一轮训练都会更新模型参数
def train(model, optimizer, criterion, train_loader, epoch):
    running_loss = 0.0  # 用于累加每个小批次的损失
    running_total = 0  # 用于累加总的样本数量
    running_correct = 0  # 用于累加预测正确的样本数量
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data  # 获取输入数据和标签
        optimizer.zero_grad()  # 清空之前的梯度

        # forward + backward + update
        outputs = model(inputs)  # 前向传播，得到模型输出
        loss = criterion(outputs, target)  # 计算损失

        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新参数

        # 累加损失
        running_loss += loss.item()

        # 计算预测准确度
        _, predicted = torch.max(outputs.data, dim=1)  # 获取每个样本的预测标签
        running_total += inputs.shape[0]  # 累加当前批次的样本数量
        running_correct += (predicted == target).sum().item()  # 累加正确预测的数量

        # 每300个小批次打印一次平均损失和准确率
        if batch_idx % 300 == 299:
            print('[%d, %5d]: loss: %.3f , acc: %.2f %%'
                  % (epoch + 1, batch_idx + 1, running_loss / 300, 100 * running_correct / running_total))
            running_loss = 0.0  # 重置损失
            running_total = 0  # 重置总样本数量
            running_correct = 0  # 重置正确预测数量

# 测试函数 ---------------------------------------------------------------------------------------
def test(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():  # 测试时不需要计算梯度
        for data in test_loader:
            images, labels = data  # 获取测试集的图像和标签
            outputs = model(images)  # 前向传播，得到输出
            _, predicted = torch.max(outputs.data, dim=1)  # 获取预测标签
            total += labels.size(0)  # 累加总样本数
            correct += (predicted == labels).sum().item()  # 累加预测正确的数量
    acc = correct / total  # 计算准确率
    return acc  # 返回准确率

# 使用Optuna进行超参数调优 ------------------------------------------------------------------------------------

# Optuna的目标函数，每次调用时会得到一组超参数，并返回模型的测试准确率
def objective(trial):
    # 使用Optuna建议的超参数范围
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)  # 使用log均匀分布选择学习率
    momentum = trial.suggest_uniform('momentum', 0.4, 0.9)  # 使用均匀分布选择动量

    # 创建模型和优化器
    model = Net()  # 创建模型
    optimizer = create_optimizer(model, learning_rate, momentum)  # 创建优化器
    criterion = torch.nn.CrossEntropyLoss()  # 选择损失函数

    acc_list_test = []  # 存储每轮测试的准确率
    for epoch in range(EPOCH):
        train(model, optimizer, criterion, train_loader, epoch)  # 训练模型
        acc_test = test(model, test_loader)  # 测试模型
        acc_list_test.append(acc_test)  # 保存测试准确率

    # 返回模型在测试集上的平均准确率，用于优化目标
    return np.mean(acc_list_test)

# 创建Optuna研究对象，并开始超参数调优
study = optuna.create_study(direction='maximize')  # 设置目标是最大化准确率
study.optimize(objective, n_trials=50)  # 进行50次试验，优化超参数

# 打印出最佳的超参数组合
print("Best hyperparameters: ", study.best_params)
