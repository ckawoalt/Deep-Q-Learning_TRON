from tron.player import Player, Direction
from tron.game import Tile, Game, PositionPlayer
from tron.window import Window
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
#from ais.basic.ai import Ai as AiBasic

import os

# General parameters
folderName = 'survivor'

# Net parameters
BATCH_SIZE = 128
GAMMA = 0.9 # Discount factor

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.batch_size = BATCH_SIZE
		self.gamma = GAMMA
		# self.conv1=nn.Conv2d(1, 8, 7,padding=3)
		# self.conv2 = nn.Conv2d(8, 32, 5, padding=2)
		# self.conv3 = nn.Conv2d(32, 128, 5,padding=2,stride=2)
		# self.conv4 = nn.Conv2d(128, 256, 3, padding=1)

		self.conv1=nn.Conv2d(1, 32, 6)
		self.conv2 = nn.Conv2d(32, 64, 3)



		self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.dropout=nn.Dropout(p=0.2)
		#self.batchnorm=nn.BatchNorm2d(1)

		self.relu=nn.ReLU()
		# self.fc1 = nn.Linear(128 * 1 * 1, 256)
		# self.fc1 = nn.Linear(512, 1024)
		# self.fc2 = nn.Linear(256, 512)
		# self.fc3 = nn.Linear(512, 256)
		# self.fc4 = nn.Linear(256, 64)
		# self.fc5 = nn.Linear(64, 4)

		self.fc1 = nn.Linear(64*5*5, 256)
		self.fc2 = nn.Linear(256, 4)

		# torch.nn.init.xavier_uniform_(self.fc0.weight)
		# torch.nn.init.xavier_uniform_(self.fc1.weight)
		# torch.nn.init.xavier_uniform_(self.fc2.weight)
		# torch.nn.init.xavier_uniform_(self.fc3.weight)
		# torch.nn.init.xavier_uniform_(self.fc4.weight)
		# torch.nn.init.xavier_uniform_(self.fc5.weight)

	def forward(self, x):

		x=x.cuda()

		# id = x
		# x=self.batchnorm(x)
		# x = self.relu(self.conv1(x))
		# x = self.conv2(x)
		# x=x+id
		# self.relu(x)
		# x = self.relu(self.conv3(x))
		# x = self.maxPool(x)

		# x = x.view(-1, 128*1*1)

		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		x = x.view(-1, 64 * 5 * 5)

		# x = self.dropout(self.relu(self.fc0(x)))
		# x = self.dropout(self.relu(self.fc1(x)))
		# x = self.dropout(self.relu(self.fc2(x)))
		# x = self.dropout(self.relu(self.fc3(x)))
		# x = self.dropout(self.relu(self.fc4(x)))
		# x = self.fc5(x)

		x=self.relu(self.fc1(x))
		x = self.fc2(x)


		return x


class Ai(Player):

	def __init__(self,epsilon=0):
		super(Ai, self).__init__()
		self.net = Net().to('cuda')
		self.epsilon = epsilon
		# Load network weights if they have been initialized already
		if os.path.isfile('ais/' + folderName + '/_ai.bak'):
			print("?")
			self.net.load_state_dict(torch.load('ais/' + folderName + '/_ai.bak'))
			#print("load reussi 1 ")
			
		elif os.path.isfile(self.find_file('_ai.bak')):
			self.net.load_state_dict(torch.load(self.find_file('_ai.bak')))
			#print("load reussi 2 ")

	def action(self, map, id):

		game_map = map.state_for_player(id)

		input = np.reshape(game_map, (1, 1, game_map.shape[0], game_map.shape[1]))
		input = torch.from_numpy(input).float()
		output = self.net(input)

		_, predicted = torch.max(output.data, 1)
		predicted = predicted.cpu().numpy()

		next_action = predicted[0] + 1

		if random.random() <= self.epsilon:
			next_action = random.randint(1,4)

		if next_action == 1:
			next_direction = Direction.UP
		if next_action == 2:
			next_direction = Direction.RIGHT
		if next_action == 3:
			next_direction = Direction.DOWN
		if next_action == 4:
			next_direction = Direction.LEFT

		return next_direction

