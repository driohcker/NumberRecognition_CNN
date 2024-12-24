import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import optuna  # 用于超参数优化的库

# 检查CUDA是否可用，如果可用，则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # 输出正在使用的设备（GPU或CPU）

# 超参数设置 ------------------------------------------------------------------------------------
batch_size = 64  # 每个训练批次的大小
EPOCH = 10  # 训练的总轮数

# 数据集准备 ------------------------------------------------------------------------------------
# 对数据进行预处理：将图片转化为Tensor，并进行标准化处理（均值0.1307，标准差0.3081）
# ToTensor()：将PIL图像或numpy.ndarray转换为tensor
# Normalize(mean, std)：标准化操作，mean和std分别是训练数据集的均值和标准差
# (0.1307, 0.3081) 是MNIST数据集的平均值和标准差，用于图像的标准化
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# 加载MNIST训练集和测试集
train_dataset = datasets.MNIST(root='./data/mnist', train=True, transform=transform)  # 训练集
test_dataset = datasets.MNIST(root='./data/mnist', train=False, transform=transform)  # 测试集

# 将数据集封装成DataLoader对象，方便按批次加载数据
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 训练数据加载器，乱序
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 测试数据加载器，按顺序加载

# 定义神经网络模型 -----------------------------------------------------------------------------------------
# 使用卷积神经网络进行分类，继承自torch.nn.Module
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 第一层卷积+ReLU激活+池化：输入1通道，输出10通道，卷积核大小5，激活函数ReLU，最大池化层2x2
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        # 第二层卷积+ReLU激活+池化：输入10通道，输出20通道，卷积核大小5，激活函数ReLU，最大池化层2x2
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        # 定义全连接层，输入320维，输出50维，接着输出10维（对应10个分类）
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)  # 获取输入数据的batch size
        x = self.conv1(x)  # 第一个卷积层
        x = self.conv2(x)  # 第二个卷积层
        x = x.view(batch_size, -1)  # 将卷积层输出展平为一维数据，适应全连接层输入
        x = self.fc(x)  # 通过全连接层
        return x  # 返回最终的输出

# 创建优化器 ---------------------------------------------------------------------------------------
# 通过给定学习率和动量参数创建SGD优化器
def create_optimizer(model, learning_rate, momentum):
    return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# 训练函数 ---------------------------------------------------------------------------------------
# 训练一个epoch的函数，更新模型参数并计算损失和准确率
def train(model, optimizer, criterion, train_loader, epoch):
    running_loss = 0.0  # 记录当前epoch的总损失
    running_total = 0  # 记录当前epoch的总样本数
    running_correct = 0  # 记录当前epoch的正确预测数

    # 迭代训练数据
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)  # 将输入数据和目标标签移到GPU上

        optimizer.zero_grad()  # 清空梯度

        # 前向传播，计算模型的输出
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, target)

        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数

        running_loss += loss.item()  # 累加损失

        # 计算当前batch的预测准确率
        _, predicted = torch.max(outputs.data, dim=1)  # 获取最大预测值的索引
        running_total += inputs.shape[0]  # 当前batch的样本数量
        running_correct += (predicted == target).sum().item()  # 当前batch正确预测的数量

        # 每300个batch输出一次平均损失和准确率
        if batch_idx % 300 == 299:
            print('[%d, %5d]: loss: %.3f , acc: %.2f %%'
                  % (epoch + 1, batch_idx + 1, running_loss / 300, 100 * running_correct / running_total))
            running_loss = 0.0  # 重置损失
            running_total = 0  # 重置总样本数
            running_correct = 0  # 重置正确预测数

# 测试函数 ---------------------------------------------------------------------------------------
# 计算模型在测试集上的准确率
def test(model, test_loader, epoch):
    correct = 0  # 正确预测的样本数
    total = 0  # 总样本数
    with torch.no_grad():  # 不计算梯度，节省内存
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # 将数据移到GPU
            outputs = model(images)  # 获取模型的输出
            _, predicted = torch.max(outputs.data, dim=1)  # 获取最大值的索引，即预测结果
            total += labels.size(0)  # 累加总样本数
            correct += (predicted == labels).sum().item()  # 累加正确预测的数量
    acc = correct / total  # 计算准确率
    print('[%d / %d]: Accuracy on test set: %.1f %% ' % (epoch+1, EPOCH, 100 * acc))  # 输出当前epoch的准确率
    return acc  # 返回准确率

# Optuna超参数调优 ------------------------------------------------------------------------------------
# 定义优化目标函数，用于超参数调优
def objective(trial):
    # 从Optuna中获取要优化的超参数
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)  # 对学习率进行对数均匀分布的搜索
    momentum = trial.suggest_uniform('momentum', 0.4, 0.9)  # 对动量进行均匀分布的搜索

    model = Net().to(device)  # 创建模型并移到GPU
    optimizer = create_optimizer(model, learning_rate, momentum)  # 创建优化器
    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数

    acc_list_test = []  # 用于保存测试集准确率的列表
    # 训练并测试模型，返回测试集的平均准确率
    for epoch in range(EPOCH):
        train(model, optimizer, criterion, train_loader, epoch)  # 训练模型
        acc_test = test(model, test_loader, epoch)  # 测试模型
        acc_list_test.append(acc_test)  # 保存每个epoch的测试集准确率

    # 绘制每轮测试准确率的曲线
    plt.plot(acc_list_test)
    plt.xlabel('Epoch')  # x轴标签
    plt.ylabel('Accuracy On TestSet')  # y轴标签
    plt.show()  # 显示图形

    return np.mean(acc_list_test)  # 返回测试集准确率的平均值，用于Optuna的优化

# 创建Optuna研究，最大化目标函数的值（即测试集准确率）
study = optuna.create_study(direction='maximize')
# 开始优化，进行1次试验
study.optimize(objective, n_trials=1)

# 输出最佳的超参数
print("Best hyperparameters: ", study.best_params)
