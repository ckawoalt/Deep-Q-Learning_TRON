import torch.nn as nn
import torch.nn.functional as F
from config import *


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 3, padding=1)

        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)

        self.pool = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)

        self.conv7 = nn.Conv2d(64, 64, 7, padding=3, stride=2)

        self.fc1 = nn.Linear(64 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)

        self.actor1 = nn.Linear(128, 64)
        self.actor2 = nn.Linear(64, 4)

        self.dropout = nn.Dropout(p=0.2)
        self.activation = self.mish

    def forward(self, x):
        '''신경망 순전파 계산을 정의'''
        x = x.to(device)

        x = self.activation(self.conv1(x))

        idx = x

        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x) + idx)

        x = self.activation(self.conv4(x))

        idx = x

        x = self.activation(self.conv5(x))
        x = self.activation(self.conv6(x) + idx)

        x = self.pool(x)

        x = self.activation(self.conv7(x))

        x = x.view(-1, 64 * 3 * 3)

        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))

        actor_output = self.actor2(self.activation(self.actor1(x)))


        return  actor_output
    def act(self, x):
        output = self(x)
        return torch.argmax(output, dim=1)

