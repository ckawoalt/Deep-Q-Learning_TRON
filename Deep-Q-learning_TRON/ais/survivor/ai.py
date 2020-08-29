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
		self.conv1 = torch.nn.Sequential(
			torch.nn.Conv2d(1, 4, kernel_size=5, stride=2, padding=2),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=3, stride=2))

		self.conv2 = torch.nn.Sequential(
			torch.nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU())

		self.conv3 = torch.nn.Sequential(
			torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU())

		self.conv4 = torch.nn.Sequential(
			torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=2, stride=1))

		# self.conv1 = nn.Conv2d(1, 8, 6,padding=3,stride=3)
		# self.conv2 = nn.Conv2d(8, 32, 3)
		# self.conv3 = nn.Conv2d(32, 64, 3)

		self.batch_norm = nn.BatchNorm2d(1)
		self.dropout = nn.Dropout(0.4)

		self.fc0 = nn.Linear(64, 128)
		self.fc1 = nn.Linear(128, 256)
		self.fc2 = nn.Linear(256, 512, bias=True)
		self.fc3 = nn.Linear(512, 512, bias=True)
		self.fc4 = nn.Linear(512, 256, bias=True)
		self.fc5 = nn.Linear(256, 128, bias=True)
		self.fc6 = nn.Linear(128, 64, bias=True)
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

		# x = self.dropout(F.relu(self.conv1(normalized)))
		# x = self.dropout(F.relu(self.conv2(x)))
		# x = self.dropout(F.relu(self.conv3(x)))
		# x = self.dropout(F.relu(self.conv4(x)))

		x = self.dropout(F.relu(self.conv1(normalized)))
		x = self.dropout(self.conv2(x))

		x = self.dropout(self.conv3(x))
		x = self.dropout(self.conv4(x))

		x = x.view(-1, 64)

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


class Ai(Player):

	def __init__(self,epsilon=0):
		super(Ai, self).__init__()
		self.net = Net()
		self.epsilon = epsilon
		# Load network weights if they have been initialized already
		if os.path.isfile('ais/' + folderName + '/ai.bak'):
			print("?")
			self.net.load_state_dict(torch.load('ais/' + folderName + '/ai.bak'))
			#print("load reussi 1 ")
			
		elif os.path.isfile(self.find_file('ai.bak')):
			self.net.load_state_dict(torch.load(self.find_file('ai.bak')))
			#print("load reussi 2 ")

	def action(self, map, id):

		game_map = map.state_for_player(id)

		input = np.reshape(game_map, (1, 1, game_map.shape[0], game_map.shape[1]))
		input = torch.from_numpy(input).float()
		output = self.net(input)

		_, predicted = torch.max(output.data, 1)
		predicted = predicted.numpy()
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

