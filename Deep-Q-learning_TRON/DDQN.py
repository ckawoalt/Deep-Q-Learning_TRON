from tron.DDQN_player import Player, Direction
from tron.DDQN_game import Tile, Game, PositionPlayer
from tron.window import Window
from collections import namedtuple,deque
from torch.utils.tensorboard import SummaryWriter
from tron.minimax import MinimaxPlayer

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
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Net parameters
BATCH_SIZE = 128
GAMMA = 0.9 # Discount factor

# Exploration parameters
EPSILON_START = 1
ESPILON_END = 0.003
DECAY_RATE = 0.999
TAU = 1e-3

# Map parameters
MAP_WIDTH = 10
MAP_HEIGHT = 10

# Memory parameters
MEM_CAPACITY = int(1e5)

# Cycle parameters
UPDATE_EVERY = 4
GAME_CYCLE = 20
DISPLAY_CYCLE = GAME_CYCLE

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA

        self.conv1=nn.Conv2d(1, 32, 6)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout=nn.Dropout(p=0.2)

        self.relu=nn.ReLU()

        self.fc1 = nn.Linear(64*5*5, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x=x.cuda()

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1, 64 * 5 * 5)

        x=self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class Agent(Player):
    def __init__(self):
        super(Agent, self).__init__()

        """Initialize an Agent object.

               Params
               =======
                   state_size (int): dimension of each state
                   action_size (int): dimension of each action
                   seed (int): random seed
               """
        # Q- Network
        self.qnetwork_local = Net().cuda()
        self.qnetwork_target = Net().cuda()
        self.action_size=4
        # if os.path.isfile('ais/' + folderName  +'/'+ '_ai.bak'):
        #     self.net.load_state_dict(torch.load('ais/' + folderName +'/' + '_ai.bak'))

        self.optimizer = optim.Adam(self.qnetwork_local.parameters())
        self.epsilon=0
        # Replay memory
        self.memory = ReplayBuffer(4, MEM_CAPACITY, BATCH_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0


    def step(self, state, action, reward, next_step, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_step, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get radom subset and learn

            if len(self.memory) > BATCH_SIZE:
                experience = self.memory.sample()
                self.learn(experience, GAMMA)


    def action(self,game_map):

        """Returns action for given state as per current policy
        Params
        =======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """


        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(game_map)
        self.qnetwork_local.train()

        # Epsilon -greedy action selction

        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy()+1)
        else:
            return random.choice(np.arange(self.action_size)+1)

    def find_file(self, name):
        return '/'.join(self.__module__.split('.')[:-1]) + '/' + name

    def get_direction(self,next_action):
        if next_action == 1:
            next_direction = Direction.UP
        if next_action == 2:
            next_direction = Direction.RIGHT
        if next_action == 3:
            next_direction = Direction.DOWN
        if next_action == 4:
            next_direction = Direction.LEFT

        return next_direction

    def next_position_and_direction(self, current_position,action):

        direction = self.get_direction(action)

        return (self.next_position(current_position, direction), direction)


    def next_position(self, current_position, direction):

        if direction == Direction.UP:
            return (current_position[0] - 1, current_position[1])
        if direction == Direction.RIGHT:
            return (current_position[0], current_position[1] + 1)
        if direction == Direction.DOWN:
            return (current_position[0] + 1, current_position[1])
        if direction == Direction.LEFT:
            return (current_position[0], current_position[1] - 1)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        =======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_state, dones = experiences
        print(experiences)
        criterion = torch.nn.MSELoss()
        self.qnetwork_local.train()
        self.qnetwork_target.eval()
        # shape of output from the model (batch_size,action_dim) = (64,4)

        predicted_targets = self.qnetwork_local(states).gather(1, actions)

        with torch.no_grad():
            labels_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)
        print("?",labels_next)
        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels = rewards + (gamma * labels_next * (1 - dones))

        loss = criterion(predicted_targets, labels).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed -size buffe to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size,):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state","action","reward","next_state","done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experiences(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


def train(local_model,target_model):
    # writer = SummaryWriter()
    #vis = visdom.Visdom()

    # vis.close(env="main")
    # loss_plot = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='loss_tracker', legend=['loss'], showlegend=True))
    # du_plot = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='duration_tracker', legend=['duration'], showlegend=True))
    # win_plot = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='rating_tracker', legend=['win_rate'], showlegend=True))
    # test_plot = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='test', legend=['test'], showlegend=True))

    # Initialize exploration rate

    epsilon = EPSILON_START
    epsilon_temp = float(epsilon)

    # Initialize the game counter
    game_counter = 0
    move_counter = 0

    win_p1 = 0
    win_p2 = 0
    changeAi = 0
    vs_min_p1_win = 0
    minimax_game = 0
    minnimax_match = 1000
    minimam_match = 1000
    Aiset = True
    otherOpponent = True
    ai='basic'
    rate=0
    # Start training

    while True:

        # Initialize the game cycle parameters
        cycle_step = 0
        p1_victories = 0
        p2_victories = 0
        null_games = 0
        player = Agent()

        # Play a cycle of games
        while cycle_step < GAME_CYCLE:

            # Increment the counters
            game_counter += 1
            cycle_step += 1

            # Initialize the starting positions
            x1 = random.randint(0,MAP_WIDTH-1)
            y1 = random.randint(0,MAP_HEIGHT-1)
            x2 = random.randint(0,MAP_WIDTH-1)
            y2 = random.randint(0,MAP_HEIGHT-1)

            while x1==x2 and y1==y2:
                x1 = random.randint(0,MAP_WIDTH-1)
                y1 = random.randint(0,MAP_HEIGHT-1)
            # Initialize the game

            player.epsilon = epsilon


            game = Game(MAP_WIDTH,MAP_HEIGHT, [
                PositionPlayer(1, player, [x1, y1]),
                PositionPlayer(2, player, [x2, y2]),])

            # Get the initial state for each player
            old_state_p1 = game.map().state_for_player(1)
            old_state_p1 = np.reshape(old_state_p1, (1, 1, old_state_p1.shape[0], old_state_p1.shape[1]))
            old_state_p1 = torch.from_numpy(old_state_p1).float()

            old_state_p2 = game.map().state_for_player(2)
            old_state_p2 = np.reshape(old_state_p2, (1, 1, old_state_p2.shape[0], old_state_p2.shape[1]))
            old_state_p2 = torch.from_numpy(old_state_p2).float()

            done=False

            while not(done):

               # print("1")

                p1_action = player.action(old_state_p1)
                p2_action = player.action(old_state_p2)

                p1_next_state, p1_reward,p2_next_state, p2_reward,done= game.step(p1_action, p2_action)

                p1_next_state = np.reshape(p1_next_state, (1, 1, p1_next_state.shape[0], p1_next_state.shape[1]))
                p1_next_state = torch.from_numpy(p1_next_state).float()

                p2_next_state = np.reshape(p2_next_state, (1, 1, p2_next_state.shape[0], p2_next_state.shape[1]))
                p2_next_state = torch.from_numpy(p2_next_state).float()

                if done:
                    if game.winner is None:
                        null_games += 1
                        p1_reward = 0
                        p2_reward = 0

                    elif game.winner == 1:
                        p1_reward = 100
                        p2_reward = -25
                        p1_victories +=1

                    else:
                        p1_reward = -25
                        p2_reward = 100
                        p2_victories += 1

                # next_originp1=p1_next_state
                #
                # p1_next_state = np.reshape(p1_next_state, (1, 1, p1_next_state.shape[0], p1_next_state.shape[1]))
                # p1_next_state = torch.from_numpy(p1_next_state).float()
                #
                # p2_next_state = np.reshape(p2_next_state, (1, 1, p2_next_state.shape[0], p2_next_state.shape[1]))
                # p2_next_state = torch.from_numpy(p2_next_state).float()

                # p1_reward = torch.from_numpy(np.array([p1_reward], dtype=np.float32)).unsqueeze(0)
                # p2_reward = torch.from_numpy(np.array([p2_reward], dtype=np.float32)).unsqueeze(0)


                player.step(old_state_p1, p1_action, p1_reward, p1_next_state, done)
                player.step(old_state_p2, p2_action, p2_reward, p2_next_state, done)

                old_state_p1 = p1_next_state
                old_state_p2 = p2_next_state

                if done:
                    torch.save(player.qnetwork_local.state_dict(), 'checkpoint.pth')
                    break
            # Update exploration rate

        nouv_epsilon = epsilon*DECAY_RATE
        if nouv_epsilon > ESPILON_END:
            epsilon = nouv_epsilon

        if epsilon==0 and game_counter%100==0 :
            epsilon = epsilon_temp


        # model.zero_grad()

        # Compute the loss
        # target_q_values_batch = target_q_values_batch.detach()
        # loss = criterion(pred_q_values_batch,target_q_values_batch)
        # #loss=F.smooth_l1_loss(pred_q_values_batch,target_q_values_batch)
        #

        # # Update bak
        p1_winrate = p1_victories / (GAME_CYCLE)
        # Display results
        if (game_counter%DISPLAY_CYCLE)==0:

            # loss_string = str(loss)
            loss_string = loss_string[7:len(loss_string)]
            loss_value = loss_string.split(',')[0]
            print("--- Match", game_counter, "---")
            print("Average duration :", float(move_counter)/float(DISPLAY_CYCLE))
            print("Loss =", loss_value)
            print("Epsilon =", epsilon)
            print("")
            #print("Max duration :", max_du)
            print("score p1 vs p2 =", win_p1, ":", win_p2)
            print("ai state=", ai)
            p1_winrate = p1_victories / (GAME_CYCLE)
            print("mini", minnimax_match)
            #print("old", old_memory.position, "posi", len(old_memory.memory), "mem size")
            # print("new", memory.position, "posi", len(memory.memory), "mem size")

            # vis_loss = float(loss_value)
            # vis.line(X=torch.tensor([game_counter]),
            #          Y=torch.tensor([vis_loss]),
            #          win=loss_plot,
            #          update='append'
            #          )
            # vis.line(X=torch.tensor([game_counter]),
            #          Y=torch.tensor([float(move_counter) / float(DISPLAY_CYCLE)]),
            #          win=du_plot,
            #          update='append'
            #          )
            # vis.line(X=torch.tensor([game_counter]),
            #          Y=torch.tensor([p1_winrate]),
            #          win=win_plot,
            #          update='append'
            #          )
            #
            # vis.line(X=torch.tensor([game_counter]),
            #          Y=torch.tensor([under_minus_26]),
            #          win=test_plot,
            #          update='append'
            #          )
            # writer.add_scalar('loss_tracker', vis_loss, game_counter)
            # writer.add_scalar('duration_tracker', (float(move_counter) / float(DISPLAY_CYCLE)), game_counter)
            # writer.add_scalar('ration_tracker', p1_winrate, game_counter)
            # writer.add_scalar('test', under_minus_26, game_counter)

            with open('ais/' + folderName +'/'+ '/data.txt', 'a') as myfile:
                myfile.write(str(game_counter) + ', ' + str(float(move_counter)/float(DISPLAY_CYCLE)) + ', ' + loss_value + '\n')

            move_counter = 0



def main():
    #model = Net().to(device)
    local_model = Net()
    target_model = Net()

    # if os.path.isfile('ais/' + folderName + '/_ai.bak'):
    #     local_model.load_state_dict(torch.load('ais/' + folderName + '/_ai.bak'))
    train(local_model,target_model)

if __name__ == "__main__":
    main()

