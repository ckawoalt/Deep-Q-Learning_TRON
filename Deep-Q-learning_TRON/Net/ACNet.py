import torch.nn as nn
import torch.nn.functional as F
from config import *
import torchvision.models as models

class Net(nn.Module):
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


        critic_output = self.critic2(self.activation(self.critic1(x)))
        critic_output = self.critic3(self.activation(critic_output))

        return critic_output, actor_output
    def act(self, x,env_prob):
        '''상태 x로부터 행동을 확률적으로 결정'''
        if env_prob is not None:
            value, actor_output = self(x,env_prob)
        else:
            value, actor_output = self(x)
        # actor_output=torch.clamp_min_(actor_output,min=0)
        # dim=1이므로 행동의 종류에 대해 softmax를 적용
        action_probs = F.softmax(actor_output, dim=1)
        # print(action_probs,"prob")
        # print(F.log_softmax(actor_output),",log prob")

        #print(action_probs)
        action = action_probs.multinomial(num_samples=1)  # dim=1이므로 행동의 종류에 대해 확률을 계산

        # print(F.log_softmax(actor_output).gather(1,action),",select")

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

        env_prob=env_prob.unsqueeze(1).detach().cuda()

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

        self.fc_env = nn.Linear(1,64)
        self.fc_env2 = nn.Linear(64, 128)


        self.actor1 = nn.Linear(128, 32)
        self.actor2 = nn.Linear(32, 4)

        self.critic1 = nn.Linear(128, 32)
        self.critic2 = nn.Linear(32, 8)
        self.critic3 = nn.Linear(8, 1)

        self.dropout = nn.Dropout(p=0)
        self.activation = self.mish
    def forward(self, x, env_prob):
        '''신경망 순전파 계산을 정의'''
        x = x.to(device)

        env_prob = env_prob.unsqueeze(1).cuda()

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


        #print(x)
        x = self.dropout(self.activation(self.fc1(x)))
        #print(x)
        x = self.dropout(self.activation(self.fc2(x)))
        #print(x)
        # x=torch.cat([x,env_prob],1)
        env_prob = torch.tanh(self.fc_env(env_prob))
        env_prob = torch.tanh(self.fc_env2(env_prob))
        # print(env_prob)
        x = x.mul(env_prob)


        actor_output = self.actor2(self.activation(self.actor1(x)))
        #print(actor_output)
        critic_output = self.critic2(self.activation(self.critic1(x)))
        critic_output = self.critic3(self.activation(critic_output))

        return critic_output, actor_output
