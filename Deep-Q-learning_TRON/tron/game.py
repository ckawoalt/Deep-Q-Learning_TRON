from time import sleep
from enum import Enum

from tron.map import Map, Tile
from tron.player import ACPlayer
from orderedset import OrderedSet

import torch
import random
import numpy as np
import queue
from config import *
from Net.ACNet import MapNet



maptype=type(MapNet())
class SetQueue(queue.Queue):
    def _init(self, maxsize):
        self.queue = OrderedSet()

    def _put(self, item):
        self.queue.add(item)

    def _get(self):
        head = self.queue.__getitem__(0)
        self.queue.remove(head)
        return head


class Winner(Enum):
    PLAYER_ONE = 1
    PLAYER_TWO = 2


class PositionPlayer:
    def __init__(self, id, player, position):
        self.id = id
        self.player = player
        self.position = position
        self.alive = True

    def body(self):
        if self.id == 1:
            return Tile.PLAYER_ONE_BODY
        elif self.id == 2:
            return Tile.PLAYER_TWO_BODY
    def slide(self):
        if self.id == 1:
            return Tile.PLAYER_ONE_slide
        elif self.id == 2:
            return Tile.PLAYER_TWO_slide

    def head(self):
        if self.id == 1:
            return Tile.PLAYER_ONE_HEAD
        elif self.id == 2:
            return Tile.PLAYER_TWO_HEAD


class HistoryElement:
    def __init__(self, mmap, player_one_direction, player_two_direction):
        self.map = mmap
        self.player_one_direction = player_one_direction
        self.player_two_direction = player_two_direction




class Game:
    def __init__(self, width, height, pps,mode=None,slide_pram=None):

        self.width = width
        self.height = height
        mmap = Map(width, height, Tile.EMPTY, Tile.WALL)
        self.history = [HistoryElement(mmap, None, None)]
        self.pps = pps
        self.winner = None
        # self.loser_len=0
        # self.winner_len = 0
        self.next_p1 = []
        self.next_p2 = []
        self.weight=[random.randint(40,101),random.randint(40,101)]
        # self.reward = 0
        self.done = False
        self.mode=mode
        self.degree=random.randint(-30,30)
        self.slide= slide if slide_pram is None else slide_pram

        for pp in self.pps:
            self.history[-1].map[pp.position[0], pp.position[1]] = pp.head()

    def map(self):
        return self.history[-1].map.clone()

    def get_rate(self,player_num=None):

        # return ((self.degree-30)/100) ** 2
        if player_num is None:
            return -((self.degree-30)*0.6)/100
        else:
            return (-((self.degree - 30) * 0.6) / 100)-((70-self.get_weight(player_num))/100)


    def get_degree(self):


        return float(self.degree)

    def get_degree_silde(self):

        return float((-self.slide*100)*(10/6)+30)

    def change_degree(self):

        if random.random()>0.5:
            temp=self.degree+random.randint(0,3)
            self.degree=min(30,temp)

        else:
            temp=self.degree-random.randint(1,5)
            self.degree=max(-30,temp)

    def prob_map(self):

        temp = np.zeros((MAP_WIDTH + 2, MAP_HEIGHT + 2))

        for i in range(MAP_HEIGHT + 2):
            for j in range(MAP_WIDTH + 2):
                temp[i][j] = self.get_degree_silde()
        # print(temp)
        return temp
    def get_weight(self,player_num):

        return self.weight[player_num]

    def get_multy(self,player_num):

        return [self.get_degree(),self.get_weight(player_num)]
    def degree_map(self):

        temp = np.zeros((MAP_WIDTH + 2, MAP_HEIGHT + 2))

        for i in range(MAP_HEIGHT + 2):
            for j in range(MAP_WIDTH + 2):
                temp[i][j] = self.get_degree()
        return temp

    def next_frame(self, action_p1, action_p2, window=None):

        map_clone = self.map()

        action = [action_p1, action_p2]

        for pp in self.pps:
            map_clone[pp.position[0], pp.position[1]] = pp.body()

        for id, pp in enumerate(self.pps):

            if type(pp.player) == type(ACPlayer()):
                (pp.position, pp.player.direction) = pp.player.next_position_and_direction(pp.position, action[id])

                if self.mode == "ice" or self.mode =="temper":
                    if pp.position[0] >= 0 and pp.position[1] >= 0 and \
                            pp.position[0] < self.width and pp.position[1] < self.height and map_clone[pp.position[0], pp.position[1]] is Tile.EMPTY:

                        rate = self.slide if self.mode =="ice" else self.get_rate(id)

                        if random.random() <= rate:

                            if(id==0):
                                self.history[-1].player_one_direction = self.pps[0].player.direction
                            else:
                                self.history[-1].player_two_direction = self.pps[1].player.direction

                            # print("미끌")
                            map_clone[pp.position[0], pp.position[1]] = pp.slide()
                            (pp.position, pp.player.direction) = pp.player.next_position_and_direction(pp.position, action[id])

            else:

                (pp.position, pp.player.direction) = pp.player.next_position_and_direction(pp.position, id + 1,self.map())

                if self.mode=="ice" or self.mode =="temper":
                    if pp.position[0] >= 0 and pp.position[1] >= 0 and \
                            pp.position[0] < self.width and pp.position[1] < self.height and map_clone[pp.position[0], pp.position[1]] is Tile.EMPTY:
                        rate = self.slide if self.mode == "ice" else self.get_rate(id)

                        if random.random() <= rate:

                            if (id == 0):
                                self.history[-1].player_one_direction = self.pps[0].player.direction
                            else:
                                self.history[-1].player_two_direction = self.pps[1].player.direction

                            # print("미끌")
                            map_clone[pp.position[0], pp.position[1]] = pp.slide()
                            (pp.position, pp.player.direction) = pp.player.next_position_and_direction(pp.position,id + 1,self.map(),pp.player.direction)

        self.history[-1].player_one_direction = self.pps[0].player.direction
        self.history[-1].player_two_direction = self.pps[1].player.direction

        # self.change_degree()

        for (id, pp) in enumerate(self.pps):
            if pp.position[0] < 0 or pp.position[1] < 0 or \
                    pp.position[0] >= self.width or pp.position[1] >= self.height:
                pp.alive, done = False, True
                map_clone[pp.position[0], pp.position[1]] = pp.head()
            elif map_clone[pp.position[0], pp.position[1]] is not Tile.EMPTY:
                pp.alive, done = False, True
                map_clone[pp.position[0], pp.position[1]] = pp.head()
            else:
                map_clone[pp.position[0], pp.position[1]] = pp.head()
        """
        if not done and independent condition 
        get player's longest path
        
        # if not done and self.check_separated(map_clone, self.pps[0]):
        #     winner = self.get_longest_path(map_clone, self.pps[0], self.pps[1])
        #     if winner == 1:
        #         self.pps[1].alive = False
        #     elif winner == 2:
        #         self.pps[0].alive = False
        #     else:
        #         self.pps[0].alive = False
        #         self.pps[1].alive = False
        """

        self.history.append(HistoryElement(map_clone, None, None))
        self.next_p1 = self.history[-1].map.state_for_player(1)
        self.next_p2 = self.history[-1].map.state_for_player(2)

        if window:
            import pygame
            while True:
                event = pygame.event.poll()

                if event.type == pygame.NOEVENT:
                    break

                for pp in self.pps:
                    try:
                        pp.player.manage_event(event)
                    except:
                        if id == 0:
                            self.winner = 2
                        elif id == 1:
                            self.winner = 1
                        return False

        return True

    def step(self, action_p1, action_p2):

        alive_count = 0
        alive = None


        if not self.next_frame(action_p1, action_p2):
            self.done = True

            return self.next_p1, self.next_p2, self.done
        for pp in self.pps:
            if pp.alive:
                alive_count += 1
                alive = pp.id

        if alive_count <= 1:
            if alive_count == 1:
                if self.pps[0].position[0] != self.pps[1].position[0] or \
                        self.pps[0].position[1] != self.pps[1].position[1]:
                    self.winner = alive

            self.done = True

        return self.next_p1,  self.next_p2,  self.done

    def main_loop(self,model, pop=None,window=None,model2=None):

        if window:
            window.render_map(self.map())

        if (not model2):
            model2=model

        while True:
            alive_count = 0
            alive = None

            if window:
                sleep(0.3)

            map=self.map()
            with torch.no_grad():
                if type(model) ==maptype:
                    action1 = model.act(torch.cat([torch.tensor(pop(map.state_for_player(1))),torch.tensor(self.prob_map()).unsqueeze(0)],0).unsqueeze(0).float())
                else:
                    action1 = model.act(torch.tensor(pop(map.state_for_player(1))).unsqueeze(0).float(),torch.tensor([self.get_multy(0)]).to(device))

                if type(model2) == maptype:
                    action1 = model2.act(torch.cat([torch.tensor(pop(map.state_for_player(2))), torch.tensor(self.prob_map()).unsqueeze(0)],0).unsqueeze(0).float())
                else:
                    action2 = model2.act(torch.tensor(pop(map.state_for_player(2))).unsqueeze(0).float(),torch.tensor([self.get_rate()]).to(device))

                # action2 = model2.act(torch.tensor(pop(map.state_for_player(2))).unsqueeze(0).float(),torch.tensor([self.slide]).to(device))

                # action1 = model.act(torch.tensor(np.expand_dims(np.concatenate((pop(map.state_for_player(1)),np.expand_dims(np.array(self.prob_map()),axis=0)),axis=0), axis=0)).float())
                # action2 = model2.act(torch.tensor(np.expand_dims(np.concatenate((pop(map.state_for_player(2)), np.expand_dims(np.array(self.prob_map()),axis=0)), axis=0),axis=0)).float())


            if not self.next_frame(action1,action2,window):
                break

            for pp in self.pps:
                if pp.alive:
                    alive_count += 1
                    alive = pp.id

            if alive_count <= 1:
                if alive_count == 1:
                    if self.pps[0].position[0] != self.pps[1].position[0] or \
                            self.pps[0].position[1] != self.pps[1].position[1]:
                        self.winner = alive
                break

            if window:
                window.render_map(self.map())
