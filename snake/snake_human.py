"""Play snake game in GUI."""

import random

import numpy as np
import pygame as pg

from snake import Snake


class SnakeHuman(Snake):
    def __init__(self, *args, **kwargs) -> None:
        super(SnakeHuman, self).__init__(*args, **kwargs)
        # init pygame
        pg.init()
        self.pygame_display = pg.display.set_mode(
            (self.sizeX*21, self.sizeY*21), 0, 32)  # init display
        pg.display.set_caption('Snake')
        self.pygame_display.fill((255, 255, 255))  # fill with white
        self.food_x, self.food_y = self.spawn_food()
        self.grid[self.food_x][self.food_y] = -2  # set food on grid
        self.display_state()

    def run(self) -> int:
        while (not self.game_over):
            events = pg.event.get()
            for event in events:
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_LEFT:
                        self.run_single(-1, 0)
                        # print('left')
                    elif event.key == pg.K_RIGHT:
                        self.run_single(1, 0)
                        # print('right')
                    elif event.key == pg.K_DOWN:
                        self.run_single(0, 1)
                        # print('down')
                    elif event.key == pg.K_UP:
                        self.run_single(0, -1)
                        # print('up')
                    self.display_state()
        return self.score


if __name__ == "__main__":
    s = SnakeHuman()
    print(s.run())
