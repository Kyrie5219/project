import pickle

file = open('../data/sample_new', 'rb')
info = pickle.load(file)
print(info)


import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Config:
    batch_size = 64
    epoch = 10
    momentum = 0.9
    alpha = 1e-3

    print_per_step = 100


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 3*3的卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 2),  #kernel_size卷积核大小 stride卷积步长 padding特征图填充
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  #2*2的最大池化层
        )

        self.fc1 = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),  # 加快收敛速度的方法（注：批标准化一般放在全连接层后面，激活函数层的前面）
            nn.ReLU()
        )

        self.fc3 = nn.Linear(64, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class TrainProcess:

    def __init__(self):
        self.train, self.test = self.load_data()
        self.net = LeNet().to(device)
        self.criterion = nn.CrossEntropyLoss()  # 定义损失函数
        self.optimizer = optim.SGD(self.net.parameters(), lr=Config.alpha, momentum=Config.momentum)

    @staticmethod
    def load_data():
        print("Loading Data......")
        """加载MNIST数据集，本地数据不存在会自动下载"""
        train_data = datasets.MNIST(root='./data/',
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)

        test_data = datasets.MNIST(root='./data/',
                                   train=False,
                                   transform=transforms.ToTensor())

        # 返回一个数据迭代器
        # shuffle：是否打乱顺序
        train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=Config.batch_size,
                                                   shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                  batch_size=Config.batch_size,
                                                  shuffle=False)
        return train_loader, test_loader

    def train_step(self):
        steps = 0
        start_time = datetime.now()

        print("Training & Evaluating......")
        for epoch in range(Config.epoch):
            print("Epoch {:3}".format(epoch + 1))

            for data, label in self.train:
                data, label = data.to(device),label.to(device)
                self.optimizer.zero_grad()  # 将梯度归零
                outputs = self.net(data)  # 将数据传入网络进行前向运算
                loss = self.criterion(outputs, label)  # 得到损失函数
                loss.backward()  # 反向传播
                self.optimizer.step()  # 通过梯度做一步参数更新

                # 每100次打印一次结果
                if steps % Config.print_per_step == 0:
                    _, predicted = torch.max(outputs, 1)
                    correct = int(sum(predicted == label))
                    accuracy = correct / Config.batch_size  # 计算准确率
                    end_time = datetime.now()
                    time_diff = (end_time - start_time).seconds
                    time_usage = '{:3}m{:3}s'.format(int(time_diff / 60), time_diff % 60)
                    msg = "Step {:5}, Loss:{:6.2f}, Accuracy:{:8.2%}, Time usage:{:9}."
                    print(msg.format(steps, loss, accuracy, time_usage))

                steps += 1

        test_loss = 0.
        test_correct = 0
        for data, label in self.test:
            with torch.no_grad():# *************修改*******************
                data, label = data.to(device),label.to(device)
                outputs = self.net(data)
                loss = self.criterion(outputs, label)
                test_loss += loss * Config.batch_size
                _, predicted = torch.max(outputs, 1)
                correct = int(sum(predicted == label))
                test_correct += correct

        accuracy = test_correct / len(self.test.dataset)
        loss = test_loss / len(self.test.dataset)
        print("Test Loss: {:5.2f}, Accuracy: {:6.2%}".format(loss, accuracy))

        end_time = datetime.now()
        time_diff = (end_time - start_time).seconds
        print("Time Usage: {:5.2f} mins.".format(time_diff / 60.))


if __name__ == "__main__":
    print(device)
    p = TrainProcess()
    p.train_step()



"""
import pickle

file = open('../data/vocab.pickle', 'rb')
info = pickle.load(file)
print(info)
"""

"""
data = [([5318, 5318, 2094], [2207, 7471, 4638, 6716, 3332, 5318, 749, 8024, 5318, 5318, 2094], 2, 1),
        ([2595, 6760], [679, 1962, 2692, 2590, 8024, 2218, 5050, 749, 8024, 2595, 6760, 4696, 4638, 4495, 4415, 679, 6844, 511, 4636, 1394, 800, 679, 7676, 1408, 8043], 2, 2),
        ([3300, 4157], [1343, 3241, 749, 8024, 3766, 4692, 1168, 2458, 1928, 511, 4692, 6814, 122, 4638, 3198, 7313, 1348, 7392, 4638, 3300, 4157, 719, 8024, 2792, 809, 4692, 749, 1962, 719, 6820, 3300, 4157, 2753, 511, 3297, 1400, 6820, 3221, 679, 7231, 4638, 511, 3146, 860, 2595, 3683, 6772, 2487], 2, 2)]
output = open('../data/sample_new', 'wb')
pickle.dump(data, output)
output.close()
"""