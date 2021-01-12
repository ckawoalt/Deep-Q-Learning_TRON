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
    def __init__(self, width, height, pps,mode=None):

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
        # self.reward = 0
        self.done = False
        self.mode=mode
        self.degree=random.randint(-15,15)

        for pp in self.pps:
            self.history[-1].map[pp.position[0], pp.position[1]] = pp.head()

    def map(self):
        return self.history[-1].map.clone()

    def get_rate(self):

        return ((self.degree-30)/100) ** 2

    def get_degree(self):

        return self.degree

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
                temp[i][j] = self.degree
        return temp

    """ 
    
    check independent condition
    
    def check_separated(self, map_clone, p1):
        path_queue = SetQueue()
        dist_map = np.copy(map_clone.state_for_player(1))
        path_queue._put((p1.position[0] + 1, p1.position[1] + 1))

        while not path_queue.empty():
            queue_elem = path_queue._get()
            x = queue_elem[0]
            y = queue_elem[1]

            dist_map[x, y] = 5

            if dist_map[x, y - 1] == 1:
                path_queue._put((x, y - 1))
            elif dist_map[x, y - 1] == -10:
                return False
            if dist_map[x + 1, y] == 1:
                path_queue._put((x + 1, y))
            elif dist_map[x + 1, y] == -10:
                return False
            if dist_map[x, y + 1] == 1:
                path_queue._put((x, y + 1))
            elif dist_map[x, y + 1] == -10:
                return False
            if dist_map[x - 1, y] == 1:
                path_queue._put((x - 1, y))
            elif dist_map[x - 1, y] == -10:
                return False

        return True

    check player's logest path 

    def get_longest_path(self, map_clone, p1, p2):
        p1_length = self.get_length(np.copy(map_clone.state_for_player(1)), p1.position[0] + 1, p1.position[1] + 1, 0, None)
        p2_length = self.get_length(np.copy(map_clone.state_for_player(2)), p2.position[0] + 1, p2.position[1] + 1, 0, None)

        # if p2_length == -10 or p1_length < p2_length:
        if p1_length < p2_length:
            self.loser_len=p1_length
            self.winner_len=p2_length
            return 2

        elif p1_length > p2_length:

            self.loser_len=p2_length
            self.winner_len=p1_length

            return 1
        else:
            return 0

 get able longest path

    def get_length(self, map_clone, x, y, length, prev_length):

        map_clone[x, y] = 5
        l1, l2, l3, l4 = -1, -1, -1, -1
        if map_clone[x, y - 1] == 1:
            l1 = self.get_length(map_clone, x, y - 1, length + 1, prev_length)
            if l1 == -10:
                return -10
        if map_clone[x + 1, y] == 1:
            l2 = self.get_length(map_clone, x + 1, y, length + 1, prev_length)
            if l2 == -10:
                return -10
        if map_clone[x, y + 1] == 1:
            l3 = self.get_length(map_clone, x, y + 1, length + 1, prev_length)
            if l3 == -10:
                return -10
        if map_clone[x - 1, y] == 1:
            l4 = self.get_length(map_clone, x - 1, y, length + 1, prev_length)
            if l4 == -10:
                return -10

        if prev_length is not None and max(l1, l2, l3, l4) > prev_length:
            return -10

        if l1 == -1 and l2 == -1 and l3 == -1 and l4 == -1:
            return length

        return max(l1, l2, l3, l4)
"""
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

                        rate = slide if self.mode =="ice" else self.get_rate()

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
                        rate = slide if self.mode == "ice" else self.get_rate()

                        if random.random() <= rate:

                            if (id == 0):
                                self.history.append(HistoryElement(map_clone, None, self.pps[1].player.direction))
                                self.history[-1].player_one_direction = self.pps[0].player.direction
                            else:
                                self.history.append(HistoryElement(map_clone,self.pps[0].player.direction, None))
                                self.history[-1].player_two_direction = self.pps[1].player.direction

                            # print("미끌")
                            map_clone[pp.position[0], pp.position[1]] = pp.slide()
                            (pp.position, pp.player.direction) = pp.player.next_position_and_direction(pp.position,id + 1,self.map(),pp.player.direction)

        self.history[-1].player_one_direction = self.pps[0].player.direction
        self.history[-1].player_two_direction = self.pps[1].player.direction

        self.change_degree()

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

            #
            # if(pop == None):
            #     with torch.no_grad():
            #         action1 = model.act(torch.tensor(np.reshape(map.state_for_player(1), (1, 1, map.state_for_player(1).shape[0],
            #                                                                               map.state_for_player(1).shape[1]))).float(),self.get_rate())
            #         action2 = model2.act(torch.tensor(np.reshape(map.state_for_player(2), (1, 1, map.state_for_player(2).shape[0],
            #                                                                               map.state_for_player(2).shape[1]))).float(),self.get_rate())
            #
            # else:

            with torch.no_grad():
                action1 = model.act(torch.tensor(pop(map.state_for_player(1))).unsqueeze(0).float(),torch.tensor([self.get_rate()]))
                action2 = model2.act(torch.tensor(pop(map.state_for_player(2))).unsqueeze(0).float(),torch.tensor([self.get_rate()]))

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
