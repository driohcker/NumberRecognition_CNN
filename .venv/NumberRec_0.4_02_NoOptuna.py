import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F

# 超参数设置 ------------------------------------------------------------------------------------
batch_size = 64  # 每个batch的大小，即每次更新权重时使用的样本数量
learning_rate = 0.01  # 学习率，控制梯度更新的步伐
momentum = 0.5  # 动量，帮助加速SGD优化过程，减少振荡
EPOCH = 10  # 训练的总轮数

# 数据集准备 ------------------------------------------------------------------------------------
# 对数据进行预处理：将图片转化为Tensor，并进行标准化处理（均值0.1307，标准差0.3081）
# ToTensor()：将PIL图像或numpy.ndarray转换为tensor
# Normalize(mean, std)：标准化操作，mean和std分别是训练数据集的均值和标准差
# (0.1307, 0.3081) 是MNIST数据集的平均值和标准差，用于图像的标准化
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# 加载MNIST训练和测试数据集，训练集和测试集分别由train=True和train=False来标记
train_dataset = datasets.MNIST(root='./data/mnist', train=True, transform=transform)  # 本地没有就加上download=True
test_dataset = datasets.MNIST(root='./data/mnist', train=False, transform=transform)  # train=True训练集，=False测试集

# 将数据集封装成DataLoader对象，方便按批次加载数据
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 随机打乱训练数据
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 测试数据按顺序加载

# 显示部分训练数据的图片和标签（30个）
fig = plt.figure()
for i in range(30):
    plt.subplot(5, 6, i+1)  # 5行6列的子图
    plt.tight_layout()
    plt.imshow(train_dataset.train_data[i], cmap='gray', interpolation='none')  # 显示图片
    plt.title("Labels: {}".format(train_dataset.train_labels[i]))  # 显示标签
    plt.xticks([])  # 隐藏x轴的刻度
    plt.yticks([])  # 隐藏y轴的刻度
plt.show()

# 定义神经网络模型 -----------------------------------------------------------------------------------------
# 使用卷积神经网络进行分类，继承自torch.nn.Module
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 第一层卷积+ReLU激活+池化：输入1通道，输出10通道，卷积核大小5，激活函数ReLU，最大池化层2x2
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),  # 输入1个通道，输出10个通道，卷积核大小5x5
            torch.nn.ReLU(),  # 激活函数ReLU
            torch.nn.MaxPool2d(kernel_size=2),  # 最大池化，池化窗口2x2
        )
        # 第二层卷积+ReLU激活+池化：输入10通道，输出20通道，卷积核大小5，激活函数ReLU，最大池化层2x2
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),  # 输入10个通道，输出20个通道，卷积核大小5x5
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        # 定义全连接层，输入320维，输出50维，接着输出10维（对应10个分类）
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),  # 输入320个特征，输出50个特征
            torch.nn.Linear(50, 10),  # 输入50个特征，输出10个特征，对应10个数字的分类
        )

    def forward(self, x):
        batch_size = x.size(0)  # 获取输入数据的batch大小
        x = self.conv1(x)  # 第一层卷积和池化
        x = self.conv2(x)  # 第二层卷积和池化
        x = x.view(batch_size, -1)  # 将特征图展开为一维向量（320维），适应全连接层输入
        x = self.fc(x)  # 全连接层输出
        return x  # 返回分类结果

# 实例化模型
model = Net()

# Construct loss and optimizer ------------------------------------------------------------------------------
# 交叉熵损失函数（用于分类问题）
criterion = torch.nn.CrossEntropyLoss()

# 使用SGD优化器，lr为学习率，momentum为动量
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# 训练函数 ---------------------------------------------------------------------------------------
# 训练一个epoch的函数，更新模型参数并计算损失和准确率
def train(epoch):
    running_loss = 0.0  # 存储每个epoch的总损失
    running_total = 0  # 存储当前epoch处理的总样本数
    running_correct = 0  # 存储当前epoch的正确分类数量

    for batch_idx, data in enumerate(train_loader, 0):  # 遍历训练集
        inputs, target = data  # 获取当前batch的数据和标签
        optimizer.zero_grad()  # 清除上一轮的梯度

        # forward + backward + update
        outputs = model(inputs)  # 模型前向传播
        loss = criterion(outputs, target)  # 计算损失

        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新权重

        # 累加损失和准确率
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, dim=1)  # 计算当前batch的预测结果
        running_total += inputs.shape[0]
        running_correct += (predicted == target).sum().item()

        # 每300次输出一次平均损失和准确率
        if batch_idx % 300 == 299:
            print('[%d, %5d]: loss: %.3f , acc: %.2f %%'
                  % (epoch + 1, batch_idx + 1, running_loss / 300, 100 * running_correct / running_total))
            running_loss = 0.0  # 重置损失
            running_total = 0  # 重置总样本数
            running_correct = 0  # 重置正确分类数

# 测试函数 ---------------------------------------------------------------------------------------
# 计算模型在测试集上的准确率
def test():
    correct = 0  # 正确分类数
    total = 0  # 总样本数
    with torch.no_grad():  # 在测试过程中不需要计算梯度
        for data in test_loader:
            images, labels = data  # 获取测试数据和标签
            outputs = model(images)  # 进行前向传播，得到预测结果
            _, predicted = torch.max(outputs.data, dim=1)  # 获取预测的类别
            total += labels.size(0)  # 更新总样本数
            correct += (predicted == labels).sum().item()  # 统计正确分类的数量
    acc = correct / total  # 计算准确率
    print('[%d / %d]: Accuracy on test set: %.1f %% ' % (epoch+1, EPOCH, 100 * acc))  # 输出测试集准确率
    return acc  # 返回测试集准确率

# Start train and Test --------------------------------------------------------------------------------------
# 主程序入口
if __name__ == '__main__':
    acc_list_test = []  # 用于存储每轮测试的准确率
    for epoch in range(EPOCH):
        train(epoch)  # 训练模型
        acc_test = test()  # 测试模型
        acc_list_test.append(acc_test)  # 保存测试准确率

    # 绘制每轮测试准确率的曲线
    plt.plot(acc_list_test)
    plt.xlabel('Epoch')  # x轴标签
    plt.ylabel('Accuracy On TestSet')  # y轴标签
    plt.show()  # 显示图形
