import torch.nn as nn
import torch.nn.functional as F
from config import *
import torchvision.models as models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # self.inception=Inception3().cuda()
        self.conv1 = nn.Conv2d(3, 32, 5,padding=2)


        self.cond1x1 = nn.Conv2d(32, 32, 1)

        self.iconv1_1 = nn.Conv2d(32, 64, 1)
        self.iconv1_2= nn.Conv2d(64, 64, 3,padding=1)

        self.iconv2_1 = nn.Conv2d(32, 64, 1)
        self.iconv2_2 = nn.Conv2d(64, 64, 5,padding=2)

        self.iconv3_1 = nn.Conv2d(32, 64, 1)
        self.iconv3_2 = nn.Conv2d(64, 64, 7,padding=3)

        self.iconv4_1 = nn.Conv2d(32, 32, (3, 1), padding=(0, 1))
        self.iconv4_2 = nn.Conv2d(32, 32, (1, 3), padding=(1, 0))

        self.ipool=nn.AvgPool2d(kernel_size=3,padding=1,stride=1)
        self.ipool_conv=nn.Conv2d(32,32,1)

        self.pool=nn.AvgPool2d(kernel_size=2)

        self.conv6=nn.Conv2d(288,288,7,padding=3)

        self.pool2=nn.MaxPool2d(kernel_size=3,stride=2)

        self.fc1 = nn.Linear(288*3*3, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 128)

        self.actor1 = nn.Linear(128, 64)
        self.actor2 = nn.Linear(64, 4)

        self.critic1 = nn.Linear(128, 64)
        self.critic2 = nn.Linear(64, 16)
        self.critic3 = nn.Linear(16, 1)

        self.dropout = nn.Dropout(p=0.4)
        self.activation = self.mish

        # self.activation=torch.tanh
    def forward(self, x):
        '''신경망 순전파 계산을 정의'''

        x = x.to(device)
        x=self.activation(self.conv1(x))

        x_1x1=self.activation(self.cond1x1(x))

        x_iconv1=self.activation(self.iconv1_1(x))
        x_iconv1=self.activation(self.iconv1_2(x_iconv1))

        x_iconv2=self.activation(self.iconv2_1(x))
        x_iconv2=self.activation(self.iconv2_2(x_iconv2))

        x_iconv3=self.activation(self.iconv3_1(x))
        x_iconv3=self.activation(self.iconv3_2(x_iconv3))

        x_iconv4=self.activation(self.iconv4_1(x))
        x_iconv4=self.activation(self.iconv4_2(x_iconv4))

        # x_pool=self.ipool(x)
        x_pool=self.activation(self.ipool_conv(x))



        x = torch.cat([x_1x1,x_iconv1,x_iconv2,x_iconv3,x_iconv4,x_pool],1)
        x = self.pool(x)
        x = self.activation(self.conv6(x))
        x = self.pool2(x)

        # print(x.size())
        x = x.view(-1, 288*3*3)
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))
        x = self.dropout(self.activation(self.fc3(x)))
        x = self.dropout(self.activation(self.fc4(x)))

        actor_output = self.actor2(self.activation(self.actor1(x)))

        critic_output = self.critic2(self.activation(self.critic1(x)))
        critic_output = self.critic3(self.activation(critic_output))


        return critic_output, actor_output

    def act(self, x,env_prob):
        '''상태 x로부터 행동을 확률적으로 결정'''
        value, actor_output = self(x,env_prob)
        # actor_output=torch.clamp_min_(actor_output,min=0)
        # dim=1이므로 행동의 종류에 대해 softmax를 적용
        action_probs = F.softmax(actor_output, dim=1)
        action = action_probs.multinomial(num_samples=1)  # dim=1이므로 행동의 종류에 대해 확률을 계산
        return action

    def deterministic_act(self, x,env_prob):
        '''상태 x로부터 행동을 확률적으로 결정'''
        value, actor_output = self(x,env_prob)
        return torch.argmax(actor_output, dim=1)

    def get_value(self, x,env_prob):
        '''상태 x로부터 상태가치를 계산'''
        value, actor_output = self(x,env_prob)

        return value

    def evaluate_actions(self, x, actions,env_prob):
        '''상태 x로부터 상태가치, 실제 행동 actions의 로그 확률, 엔트로피를 계산'''
        value, actor_output = self(x,env_prob)

        log_probs = F.log_softmax(actor_output, dim=1)  # dim=1이므로 행동의 종류에 대해 확률을 계산
        action_log_probs = log_probs.gather(1, actions.detach())  # 실제 행동의 로그 확률(log_probs)을 구함

        probs = F.softmax(actor_output, dim=1)  # dim=1이므로 행동의 종류에 대한 계산
        entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, entropy

    def mish(self, x):
        return x * torch.tanh(F.softplus(x))


class Net2(Net):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)

        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)

        self.pool = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)

        self.conv7 = nn.Conv2d(64, 64, 7, padding=3, stride=2)

        self.fc1 = nn.Linear(64 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)

        self.actor1 = nn.Linear(129, 64)
        self.actor2 = nn.Linear(64, 4)

        self.critic1 = nn.Linear(129, 64)
        self.critic2 = nn.Linear(64, 16)
        self.critic3 = nn.Linear(16, 1)

        self.dropout = nn.Dropout(p=0.2)
        self.activation = self.mish

    def forward(self, x,env_prob):
        '''신경망 순전파 계산을 정의'''
        x = x.to(device)
        env_prob=torch.tensor(env_prob).unsqueeze(1).cuda()

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

        x=torch.cat([x,env_prob],1)

        actor_output = self.actor2(self.activation(self.actor1(x)))

        critic_output = self.critic2(self.activation(self.critic1(x)))
        critic_output = self.critic3(self.activation(critic_output))

        return critic_output, actor_output

class Net3(Net):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 3,padding=1)

        self.conv2 = nn.Conv2d(32, 32, 3,padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3,padding=1)

        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)

        self.pool=nn.AvgPool2d(kernel_size=3, padding=1, stride=2)

        self.conv7=nn.Conv2d(64,64,7,padding=3, stride=2)

        self.fc1 = nn.Linear(64*3*3, 256)
        self.fc2 = nn.Linear(256, 128)

        self.actor1 = nn.Linear(128, 64)
        self.actor2 = nn.Linear(64, 4)

        self.critic1 = nn.Linear(128, 64)
        self.critic2 = nn.Linear(64, 16)
        self.critic3 = nn.Linear(16, 1)

        self.dropout = nn.Dropout(p=0.2)
        self.activation = self.mish

    def forward(self, x):
        '''신경망 순전파 계산을 정의'''
        x = x.to(device)

        x = self.activation(self.conv1(x))

        idx = x

        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x)+idx)

        x = self.activation(self.conv4(x))

        idx = x

        x = self.activation(self.conv5(x))
        x = self.activation(self.conv6(x)+idx)

        x = self.pool(x)

        x = self.activation(self.conv7(x))

        x = x.view(-1, 64*3*3)

        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))

        actor_output = self.actor2(self.activation(self.actor1(x)))

        critic_output = self.critic2(self.activation(self.critic1(x)))
        critic_output = self.critic3(self.activation(critic_output))

        return critic_output, actor_output