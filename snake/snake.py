# snake

import random
from typing import Tuple

import numpy as np
import pygame as pg
from helper import Globe as globe


class Snake:
    def __init__(self, x_grid_size: int = 20, y_grid_size: int = 20) -> None:
        # grid
        self.grid_size_x = x_grid_size  # width of grid
        self.grid_size_y = y_grid_size  # height of grid
        # grid  -2 = food, -1 = empty, any number <= 0 is the position of the snakes body. eg if the snake has length 10 then 0 is the head and 9 is the tail
        self.grid = np.zeros((self.grid_size_x, self.grid_size_y))-1

        # snake
        self.head_x = int(1)  # snake head position x
        self.head_y = int(1)  # snake head position y
        self.length = 0  # current length of snake
        self.grid[self.head_x][self.head_y] = 0  # set snake head on the grid

        # scoring
        self.score = 0  # current score
        self.score_per_food = 100  # point modification for eating food
        self.score_per_move = -1  # point modificaiton for moving
        self.score_penalty_for_failure = -50  # point modification for dying
        self.game_over_threshold = -self.grid_size_x*self.grid_size_y*2
        self.move_threshold = self.grid_size_x*self.grid_size_y*2
        self.moves = 0

        self.moves_since_food = 0

        # input
        dir = random.randint(0, 3)
        if dir == 0:
            self.direction_x = 0  # 0,1,2,3 = up,right,down,left
            self.direction_y = -1
        elif dir == 1:
            self.direction_x = 1
            self.direction_y = 0
        elif dir == 2:
            self.direction_x = 0
            self.direction_y = -1
        else:
            self.direction_x = -1
            self.direction_y = 0

        # gamestate
        self.game_over = False

        self.food_x, self.food_y = self.spawn_food()

    def run_single(self, x_direction: int, y_direction: int) -> None:
        self.direction_x = x_direction
        self.direction_y = y_direction
        self.step_time()

    def display_state(self) -> None:
        # draw grid
        # font = pg.font.Font('freesansbold.ttf',12)
        for i in range(self.grid_size_x):
            for j in range(self.grid_size_y):
                pg.draw.rect(self.DISPLAY, self.grid_num_2_color(
                    self.grid[i][j]), (i*21, j*21, 20, 20))
                # text = font.render(str(int(self.grid[i][j])), True, (255,255,255))
                # textRect = text.get_rect()
                # textRect.center = (i*21+10,j*21+10)
                # self.DISPLAY.blit(text,textRect)
        pg.display.update()

    # TODO change num to enum
    def grid_num_2_color(self, num: int) -> Tuple[int, int, int]:
        if num == -2:  # food
            return (255, 0, 0)
        elif num == 0:  # head
            return (0, 0, 0)
        elif num > 0:  # tail
            return (100, 100, 100)
        else:  # background
            return (200, 200, 200)

    def step_time(self) -> None:
        # move head
        self.head_x += self.direction_x
        self.head_y += self.direction_y
        self.score += self.score_per_move

        # check if snake ate
        ateThisTurn = False
        if (self.head_x == self.food_x) and (self.head_y == self.food_y):
            self.length += 1
            self.food_x, self.food_y = self.spawn_food()
            self.grid[self.food_x][self.food_y] = -2
            self.score += (self.score_per_food)
            ateThisTurn = True
            self.moves_since_food = 0
            if self.length > 15:  # if snake is max length, the game has been won
                self.score += 1000
                self.game_over = True

        # move body
        for i in range(self.grid_size_x):
            for j in range(self.grid_size_y):
                if self.grid[i][j] >= 0:
                    self.grid[i][j] += 1
                    # if the gird length is longer than the actual length, delete the tail
                    if (self.grid[i][j] > self.length) and (not ateThisTurn):
                        self.grid[i][j] = -1

        # check if dead
        self.game_over = self.check_game_over()
        if self.game_over:
            self.score += self.score_penalty_for_failure

        if not self.game_over:
            # set head on grid
            self.grid[self.head_x][self.head_y] = 0

    def spawn_food(self) -> Tuple[int, int]:
        # generate mask matrix and count empty spots
        mask = np.zeros((self.grid_size_x, self.grid_size_y))
        count = 1
        for i in range(self.grid_size_x):
            for j in range(self.grid_size_y):
                if self.grid[i][j] == -1:
                    mask[i][j] = count
                    count += 1

        # generate a random number from 0-count
        if count > 1:
            spot = random.randint(1, count-1)
        else:
            return(i, j)

        # print(mask)
        #print(spot, count)

        # find the x and y location of the spot
        for i in range(self.grid_size_x):
            for j in range(self.grid_size_y):
                if (mask[i][j] == spot):
                    return (i, j)
        print('not found')
        return (i, j)

    def check_game_over(self) -> bool:
        # check if we ran into a wall
        if (self.head_x < 0) or (self.head_x >= self.grid_size_x) or self.score < self.game_over_threshold or self.moves > self.move_threshold:
            return True
        elif (self.head_y < 0) or (self.head_y >= self.grid_size_y):
            return True
        elif self.moves_since_food > globe.TIMEOUT:
            return True

        # check if we ran into the body
        if (self.grid[self.head_x][self.head_y] >= 0):
            return True

        return False
