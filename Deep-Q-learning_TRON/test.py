import gym
import random
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, seq_data):
        self.buffer.append(seq_data)

    def sample(self, on_policy=False):
        if on_policy:
            mini_batch = [self.buffer[-1]]
        else:
            mini_batch = random.sample(self.buffer, batch_size)

        s_lst, a_lst, r_lst, prob_lst, done_lst, is_first_lst = [], [], [], [], [], []
        for seq in mini_batch:
            is_first = True  # Flag for indicating whether the transition is the first item from a sequence
            for transition in seq:
                s, a, r, prob, done = transition

                s_lst.append(s)
                a_lst.append([a])
                r_lst.append(r)
                prob_lst.append(prob)
                done_mask = 0.0 if done else 1.0
                done_lst.append(done_mask)
                is_first_lst.append(is_first)
                is_first = False

        s, a, r, prob, done_mask, is_first = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                             r_lst, torch.tensor(prob_lst, dtype=torch.float), done_lst, \
                                             is_first_lst
        return s, a, r, prob, done_mask, is_first

    def size(self):
        return len(self.buffer)


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_q = nn.Linear(256, 2)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        pi = F.softmax(x, dim=softmax_dim)
        return pi

    def q(self, x):
        x = F.relu(self.fc1(x))
        q = self.fc_q(x)
        return q


def train(model, optimizer, memory, on_policy=False):
    s, a, r, prob, done_mask, is_first = memory.sample(on_policy)

    q = model.q(s)
    q_a = q.gather(1, a)
    pi = model.pi(s, softmax_dim=1)
    pi_a = pi.gather(1, a)
    v = (q * pi).sum(1).unsqueeze(1).detach()

    rho = pi.detach() / prob
    rho_a = rho.gather(1, a)
    rho_bar = rho_a.clamp(max=c)

    q_ret = v[-1] * done_mask[-1]
    q_ret_lst = []
    for i in reversed(range(len(r))):
        q_ret = r[i] + gamma * q_ret
        q_ret_lst.append(q_ret.item())
        q_ret = (rho_bar[i] + ((rho_a[i] - c) / rho_a[i]).clamp(min=0)) * (q_ret - q_a[i]) + v[i]

        if is_first[i] and i != 0:
            q_ret = v[i - 1] * done_mask[i - 1]  # When a new sequence begins, q_ret is initialized

    q_ret_lst.reverse()
    q_ret = torch.tensor(q_ret_lst, dtype=torch.float).unsqueeze(1)

    prob_a = prob.gather(1, a)
    ratio_a = torch.min(pi_a / prob_a, torch.clamp(pi_a / prob_a, 1 - clipping, 1 + clipping))
    grad = - ratio_a * (q_ret - v)  # that's all about ppo
    loss = grad.mean() + F.smooth_l1_loss(q_a, q_ret)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Hyperparameters
learning_rate = 0.0002
gamma = 0.98
buffer_limit = 6000
rollout_len = 10
batch_size = 4
c = 1.5  # For truncating importance sampling ratio
possion_lambda = 4
clipping = 0.1


def main():
    env = gym.make('CartPole-v1')
    memory = ReplayBuffer()
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s = env.reset()
        done = False

        while not done:
            seq_data = []
            for t in range(rollout_len):
                prob = model.pi(torch.from_numpy(s).float())
                a = Categorical(prob).sample().item()
                s_prime, r, done, info = env.step(a)
                seq_data.append((s, a, r / 100.0, prob.detach().numpy(), done))

                score += r
                s = s_prime
                if done:
                    break

            memory.put(seq_data)
            if memory.size() > 500:
                train(model, optimizer, memory, on_policy=True)
                for i in range(np.random.poisson(possion_lambda)):
                    train(model, optimizer, memory)

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}, buffer size : {}".format(n_epi, score / print_interval,
                                                                                  memory.size()))
            score = 0.0

    env.close()


if __name__ == '__main__':
    main()