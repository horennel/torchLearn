import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class LeNet(nn.Module):
    """
    输入为：[1, 28, 28]
    第一层：卷积[6, 5, 5]，填充2，步幅1
           平均池化[2, 2]，填充0，步幅2
    第二层：卷积[6, 5, 5]，填充0，步幅1
           平均池化[2, 2]，填充0，步幅2
    第三层：全连接神经元120个
    第四层：全连接神经元84个
    第五层：全连接神经元10个
    """

    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # 2x2 池化层
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 16 x 5 x 5 -> 120
        self.fc2 = nn.Linear(120, 84)  # 120 -> 84
        self.fc3 = nn.Linear(84, num_classes)  # 84 -> 10 (分类)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Conv1 + ReLU
        x = self.pool(x)  # Pool1
        x = F.relu(self.conv2(x))  # Conv2 + ReLU
        x = self.pool(x)  # Pool2
        x = torch.flatten(x, 1)  # 展平为全连接层输入
        x = F.relu(self.fc1(x))  # FC1 + ReLU
        x = F.relu(self.fc2(x))  # FC2 + ReLU
        x = self.fc3(x)  # 输出层
        return x


if __name__ == '__main__':
    model = LeNet()
    print(summary(model, (1, 28, 28)))
