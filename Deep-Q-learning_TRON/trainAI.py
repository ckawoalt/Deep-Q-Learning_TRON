from tron.player import Player, Direction
from tron.game import Tile, Game, PositionPlayer
from tron.window import Window
from collections import namedtuple
from tron.minimax import MinimaxPlayer
from torch.utils.tensorboard import SummaryWriter
from tron import resnet

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import visdom
import os


# General parameters
folderName = 'survivor'

# Net parameters
BATCH_SIZE = 256
GAMMA = 0.9  # Discount factor

# Exploration parameters
EPSILON_START = 1
ESPILON_END = 0.03
DECAY_RATE = 0.999

# Map parameters
MAP_WIDTH = 10
MAP_HEIGHT = 10
writer=SummaryWriter()
# Memory parameters
MEM_CAPACITY = 15000
Bottleneck=resnet.Bottleneck
BasicBlock=resnet.BasicBlock
conv1x1=resnet.conv1x1
conv3x3=resnet.conv3x3

# Cycle parameters
GAME_CYCLE = 30
DISPLAY_CYCLE = GAME_CYCLE
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA



        #resnet 이전
        self.conv1 = torch.nn.Sequential(

            torch.nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1,bias=False),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2))


        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1,bias=False),
            torch.nn.ReLU())

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU())

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3,stride=1))

        #가장 초기
        # self.conv1 = nn.Conv2d(1, 8, 6,padding=3,stride=3)
        #  self.conv2 = nn.Conv2d(8, 32, 3)
        #  self.conv3 = nn.Conv2d(32, 64, 3)
        self.batch_norm = nn.BatchNorm2d(1)


        self.dropout = nn.Dropout(0.4)

        self.fc0 = nn.Linear(64, 128)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512,bias=True)
        self.fc3 = nn.Linear(512, 512,bias=True)
        self.fc4 = nn.Linear(512, 256,bias=True)
        self.fc5 = nn.Linear(256, 128, bias=True)
        self.fc6 = nn.Linear(128, 64,bias=True)
        self.fc7 = nn.Linear(64, 32, bias=True)
        self.fc8 = nn.Linear(32, 4, bias=True)


        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        torch.nn.init.xavier_uniform_(self.fc5.weight)
        torch.nn.init.xavier_uniform_(self.fc6.weight)
        torch.nn.init.xavier_uniform_(self.fc7.weight)
        torch.nn.init.xavier_uniform_(self.fc8.weight)
    def forward(self, x):

        normalized = self.batch_norm(x)

        # x = self.conv1(x)
        # x = self.dropout(F.relu(self.conv1(x)))
        # x = self.dropout(F.relu(self.conv2(x)))
        # x = self.dropout(F.relu(self.conv3(x)))
        # x = self.dropout(F.relu(self.conv4(x)))

        x = self.dropout(F.relu(self.conv1(normalized)))
        x = self.dropout(self.conv2(x))
        x = self.dropout(self.conv3(x))
        x = self.dropout(self.conv4(x))

        #x = self.conv1(x)
        #x = self.conv2(x)
        #x = self.conv3(x)
        #x = self.conv4(x)

        x = x.view(-1,64)

        x = self.dropout(F.relu(self.fc0(x)))
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.dropout(F.relu(self.fc5(x)))
        x = self.dropout(F.relu(self.fc6(x)))
        x = self.dropout(F.relu(self.fc7(x)))
        x = self.fc8(x)

        return x

class ResNet(nn.Module):



    def __init__(self, block, layers, num_classes=4, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.gamma = GAMMA
        self.inplanes = 16
        self.batch_size = BATCH_SIZE
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)


        self.bn1 = nn.BatchNorm2d(16)

        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout=nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024,800)
        self.fc3 = nn.Linear(800,600)
        self.fc4 = nn.Linear(600,512)
        self.fc5 = nn.Linear(512,256)
        self.fc6 = nn.Linear(256,128)
        self.fc7 = nn.Linear(128,64)
        self.fc8 = nn.Linear(64,4)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        torch.nn.init.xavier_uniform_(self.fc5.weight)
        torch.nn.init.xavier_uniform_(self.fc6.weight)
        torch.nn.init.xavier_uniform_(self.fc7.weight)
        torch.nn.init.xavier_uniform_(self.fc8.weight)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x=x.cuda()
        x = self.conv1(x)
        # x.shape =[1, 16, 32,32]
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        # x.shape =[1, 128, 32,32]
        x = self.layer2(x)
        # x.shape =[1, 256, 32,32]
        x = self.layer3(x)
        # x.shape =[1, 512, 16,16]
        x = self.layer4(x)
        # x.shape =[1, 1024, 8,8]

        x = self.avgpool(x)

        x = x.view(-1,256)

       # print(x.shape)
        x = self.dropout(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.dropout(F.relu(self.fc5(x)))
        x = self.dropout(F.relu(self.fc6(x)))
        x = self.dropout(F.relu(self.fc7(x)))
        x = self.fc8(x)

        return x
class Ai(Player):

    def __init__(self, epsilon=0):
        super(Ai, self).__init__()
        self.net = Net().to(device)
        #self.net = ResNet(resnet.Bottleneck, [3, 4, 6, 3], 10, True).to(device)

        self.epsilon = epsilon
        # Load network weights if they have been initialized already
        if os.path.isfile('ais/' + folderName + '/ai.bak'):
            self.net.load_state_dict(torch.load('ais/' + folderName + '/ai.bak'))
        # print("load reussi 1 ")

        elif os.path.isfile(self.find_file('ai.bak')):
            self.net.load_state_dict(torch.load(self.find_file('ai.bak')))
        # print("load reussi 2 ")

    def action(self, map, id):

        game_map = map.state_for_player(id)

        input = np.reshape(game_map, (1, 1, game_map.shape[0], game_map.shape[1]))
        input = torch.from_numpy(input).float()
        output = self.net(input)

        #print(output)
        _, predicted = torch.max(output.data, 1)
        predicted=predicted.cpu()
        predicted = predicted.numpy()
        next_action = predicted[0] + 1

        if random.random() <= self.epsilon:
            next_action = random.randint(1, 4)

        if next_action == 1:
            next_direction = Direction.UP
        if next_action == 2:
            next_direction = Direction.RIGHT
        if next_action == 3:
            next_direction = Direction.DOWN
        if next_action == 4:
            next_direction = Direction.LEFT
        #print(next_direction)
        return next_direction


Transition = namedtuple('Transition', ('old_state', 'action', 'new_state', 'reward', 'terminal'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.p=0.8

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def push_old(self,i):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] =i
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def delete_old(self,batch_size):

        del_mem=random.sample(self.memory, batch_size)
        save_mem=random.sample(del_mem,int(len(del_mem)*self.p))

        self.memory=list(set(self.memory) - set(save_mem))
        self.position=len(self.memory)
        return del_mem
    def thanos(self):

        self.memory = random.sample(self.memory, int(len(self.memory)*0.3))
        self.position=len(self.memory)

    def __len__(self):
        return len(self.memory)


def train(model):
    # Initialize neural network parameters and optimizer
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss().to(device)
    max_du = 0
    win_p1=0
    win_p2=0
    changeAi=0
    vs_min_p1_win=0
    minimax_game=0
    minnimax_match=1000
    minimam_match=1000
    Aiset=True
    learnAi=True
    rate=0

    # Initialize exploration rate
    epsilon = EPSILON_START
    epsilon_temp = float(epsilon)

    # Initialize memory
    memory = ReplayMemory(MEM_CAPACITY)
    old_memory=ReplayMemory(MEM_CAPACITY)

    # Initialize the game counter
    game_counter = 0
    move_counter = 0

    vis = visdom.Visdom()
    vis.close(env="main")
    loss_plot = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='loss_tracker', legend=['loss'], showlegend=True))
    du_plot = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='duration_tracker', legend=['duration'], showlegend=True))
    win_plot = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='rating_tracker', legend=['win_rate'], showlegend=True))
    test_plot = vis.line(Y=torch.Tensor(1).zero_(),  opts=dict(title='test', legend=['test'], showlegend=True))
    # Start training

    player_2 = Ai(epsilon)
    ai = 'basic Ai'
    while True:

        # Initialize the game cycle parameters
        cycle_step = 0
        p1_victories = 0
        p2_victories = 0
        null_games = 0
        player_1 = Ai(epsilon)
        if(learnAi):
            player_2 = Ai(epsilon)

        # Play a cycle of games
        while cycle_step < GAME_CYCLE:

            changeAi += 1

            if(game_counter<15000):
                changeAi=0

            if (changeAi > minnimax_match):

                if (Aiset):
                    player_2 = Ai(epsilon)
                    player_2.epsilon = epsilon
                    player_2 = MinimaxPlayer(2, 'VORNOI')

                    Aiset=False
                    learnAi=False

                    ai = 'minimax'
                    minnimax_match = (10000 * rate) + minimam_match

                else:
                    rate=vs_min_p1_win/minimax_game
                    minnimax_match = 10000 -minnimax_match

                    vs_min_p1_win = 0
                    minimax_game = 0
                    learnAi = True
                    Aiset = True
                    player_2 = Ai(epsilon)
                    player_2.epsilon = epsilon
                    ai = 'basic Ai'
                changeAi = 0

            # Increment the counters
            game_counter += 1
            cycle_step += 1

            # Initialize the starting positions
            x1 = random.randint(0, MAP_WIDTH - 1)
            y1 = random.randint(0, MAP_HEIGHT - 1)
            x2 = random.randint(0, MAP_WIDTH - 1)
            y2 = random.randint(0, MAP_HEIGHT - 1)
            while x1 == x2 and y1 == y2:
                x1 = random.randint(0, MAP_WIDTH - 1)
                y1 = random.randint(0, MAP_HEIGHT - 1)

            # Initialize the game
            player_1.epsilon = epsilon
            if(learnAi):
                player_2 = Ai(epsilon)
                player_2.epsilon = epsilon

            game = Game(MAP_WIDTH, MAP_HEIGHT, [
                PositionPlayer(1, player_1, [x1, y1]),
                PositionPlayer(2, player_2, [x2, y2]), ])

            # Get the initial state for each player
            old_state_p1 = game.map().state_for_player(1)
            old_state_p1 = np.reshape(old_state_p1, (1, 1, old_state_p1.shape[0], old_state_p1.shape[1]))
            old_state_p1 = torch.from_numpy(old_state_p1).float()


            old_state_p2 = game.map().state_for_player(2)
            old_state_p2 = np.reshape(old_state_p2, (1, 1, old_state_p2.shape[0], old_state_p2.shape[1]))
            old_state_p2 = torch.from_numpy(old_state_p2).float()


            # Run the game
            # if(game_counter>4000):
            #	 window = Window(game, 40)
            #	 game.main_loop(window)
            # else:
            game.main_loop()

            # Analyze the game
            move_counter += len(game.history)
            terminal = False

            for historyStep in range(len(game.history) - 1):

                # Get the state for each player
                new_state_p1 = game.history[historyStep + 1].map.state_for_player(1)
                new_state_p1 = np.reshape(new_state_p1, (1, 1, new_state_p1.shape[0], new_state_p1.shape[1]))
                new_state_p1 = torch.from_numpy(new_state_p1).float()

                new_state_p2 = game.history[historyStep + 1].map.state_for_player(2)
                new_state_p2 = np.reshape(new_state_p2, (1, 1, new_state_p2.shape[0], new_state_p2.shape[1]))
                new_state_p2 = torch.from_numpy(new_state_p2).float()

                # Get the action for each player
                if game.history[historyStep].player_one_direction is not None:
                    action_p1 = torch.from_numpy(np.array([game.history[historyStep].player_one_direction.value - 1],
                                                          dtype=np.float32)).unsqueeze(0)
                    action_p2 = torch.from_numpy(np.array([game.history[historyStep].player_two_direction.value - 1],
                                                          dtype=np.float32)).unsqueeze(0)
                else:
                    action_p1 = torch.from_numpy(np.array([0], dtype=np.float32)).unsqueeze(0)
                    action_p2 = torch.from_numpy(np.array([0], dtype=np.float32)).unsqueeze(0)

                # Compute the reward for each player
                reward_p1 = +1
                reward_p2 = +1
                # print(game.history)

                if historyStep + 1 == len(game.history) - 1:
                    if game.winner is None:
                        null_games += 1
                        reward_p1 = 0
                        reward_p2 = 0
                    elif game.winner == 1:
                        reward_p1 = 100
                        reward_p2 = -25
                        p1_victories += 1
                        if not (Aiset):
                            vs_min_p1_win+=1
                    else:
                        reward_p1 = -25
                        reward_p2 = 100
                        p2_victories += 1
                    if not (Aiset):
                        minimax_game += 1
                    terminal = True

                reward_p1 = torch.from_numpy(np.array([reward_p1], dtype=np.float32)).unsqueeze(0)
                reward_p2 = torch.from_numpy(np.array([reward_p2], dtype=np.float32)).unsqueeze(0)

                # Save the transition for each player
                memory.push(old_state_p1, action_p1, new_state_p1, reward_p1, terminal)
                memory.push(old_state_p2, action_p2, new_state_p2, reward_p2, terminal)

                # Update old state for each player
                old_state_p1 = new_state_p1
                old_state_p2 = new_state_p2

            # Update exploration rate
            nouv_epsilon = epsilon * DECAY_RATE
            if nouv_epsilon > ESPILON_END:
                epsilon = nouv_epsilon
            if epsilon == 0 and game_counter % 100 == 0:
                epsilon = espilon_temp

        # Get a sample for training

        transitions = memory.delete_old(min(len(memory), model.batch_size))

        n_transition = old_memory.sample(min(len(old_memory), max(model.batch_size - 64,model.batch_size)))

        for i in transitions:
            old_memory.push_old(i)


        transitions += n_transition
        batch = Transition(*zip(*transitions))


        old_state_batch = torch.cat(batch.old_state)
        action_batch = torch.cat(batch.action).long()
        new_state_batch = torch.cat(batch.new_state)
        reward_batch = torch.cat(batch.reward).to(device)

        # Compute predicted Q-values for each action
        pred_q_values_batch = torch.sum(model(old_state_batch.to(device)).gather(1, action_batch.to(device)), dim=1)

        pred_q_values_batch=pred_q_values_batch.to(device)

        pred_q_values_next_batch = model(new_state_batch.to(device)).to(device)

        under_minus_26 = 0
        # Compute targeted Q-value for action performed
        target_q_values_batch = torch.cat(tuple(reward_batch[i] if batch[4][i]
                                                else reward_batch[i] + model.gamma * torch.max(pred_q_values_next_batch[i])
                                                for i in range(len(reward_batch))))

        target_q_values_batch=target_q_values_batch.to(device)

        if(torch.min(pred_q_values_next_batch)<(-25)):
            if(game_counter % 100 == 0):
                under_minus_26 = torch.sum(pred_q_values_next_batch < -25).squeeze()
                #print(under_minus_26)


        win_p1 += p1_victories
        win_p2 += p2_victories
        # zero the parameter gradients
        model.zero_grad()

        # Compute the loss
        target_q_values_batch = target_q_values_batch.detach()
        loss = criterion(pred_q_values_batch, target_q_values_batch)

        # Do backward pass
        loss.backward()
        optimizer.step()
        if(old_memory.position>10000):
            old_memory.thanos()
        if(memory.position>10000):
            memory.thanos()
        # Update bak
        torch.save(model.state_dict(), 'ais/' + folderName + '/ai.bak')
        if(game_counter==10000):
            print("st")
        # Display results
        if (game_counter % DISPLAY_CYCLE) == 0:
            if (max_du < (float(move_counter) / float(DISPLAY_CYCLE))):
                max_du = float(move_counter) / float(DISPLAY_CYCLE)

            loss_string = str(loss)
            loss_string = loss_string[7:len(loss_string)]
            loss_value = loss_string.split(',')[0]
            print("--- Match", game_counter, "---")
            print("Average duration :", float(move_counter) / float(DISPLAY_CYCLE))
            print("Loss =", loss_value)
            print("Epsilon =", epsilon)
            print("Max duration :", max_du)
            print("score p1 vs p2 =",win_p1,":",win_p2)
            print(ai)
            print(changeAi)
            print("mini",minnimax_match)
            print("old",old_memory.position,"posi",len(old_memory.memory),"mem size")
            print("new",memory.position,"posi",len(memory.memory),"mem size")
            p1_winrate=p1_victories/(p1_victories+p2_victories)
            vis_loss = float(loss_value)
            vis.line(X=torch.tensor([game_counter]),
                     Y=torch.tensor([vis_loss]),
                     win=loss_plot,
                     update='append'
                     )
            vis.line(X=torch.tensor([game_counter]),
                     Y=torch.tensor([float(move_counter) / float(DISPLAY_CYCLE)]),
                     win=du_plot,
                     update='append'
                     )
            vis.line(X=torch.tensor([game_counter]),
                     Y=torch.tensor([p1_winrate]),
                     win=win_plot,
                     update='append'
                     )

            vis.line(X=torch.tensor([game_counter]),
                     Y=torch.tensor([under_minus_26]),
                     win=test_plot,
                     update='append'
                     )
            writer.add_scalar('loss_tracker',vis_loss,game_counter)
            writer.add_scalar('duration_tracker', (float(move_counter) / float(DISPLAY_CYCLE)),game_counter)
            writer.add_scalar('ration_tracker', p1_winrate,game_counter)
            writer.add_scalar('test', under_minus_26,game_counter)


            with open('ais/' + folderName + '/data.txt', 'a') as myfile:
                myfile.write(str(game_counter) + ', ' + str(
                    float(move_counter) / float(DISPLAY_CYCLE)) + ', ' + loss_value + '\n')
            move_counter = 0



def main():
    #model = ResNet(resnet.Bottleneck, [3, 4, 6, 3], 10, True).to(device)
    model= Net().to(device)

    if os.path.isfile('ais/' + folderName + '/ai.bak'):
        model.load_state_dict(torch.load('ais/' + folderName + '/ai.bak'))
    train(model)


if __name__ == "__main__":
    main()

