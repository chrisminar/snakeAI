from snake import Snake
import random
import numpy as np
import pygame as pg


class SnakeHuman(Snake):
  def __init__(self, *args, **kwargs):
    super(snakeHuman, self).__init__(*args, **kwargs)
    #init pygame
    pg.init()
    self.DISPLAY = pg.display.set_mode((self.sizeX*21,self.sizeY*21),0,32)#init display
    pg.display.set_caption('Snake')
    self.DISPLAY.fill((255,255,255)) #fill with white
    self.foodX,self.foodY = self.spawn_food()
    self.grid[self.foodX][self.foodY] = -2   #set food on grid
    self.displayState()

  def aiGameInit(self):
    self.foodX,self.foodY = self.spawn_food()
    self.grid[self.foodX][self.foodY] = -2   #set food on grid

  def run(self):
    while (self.gameover == False):
      events = pg.event.get()
      for event in events:
        if event.type==pg.KEYDOWN:
          if event.key == pg.K_LEFT:
            self.runSingle(-1, 0)
            #print('left')
          elif event.key == pg.K_RIGHT:
            self.runSingle(1,0)
            #print('right')
          elif event.key == pg.K_DOWN:
            self.runSingle(0,1)
            #print('down')
          elif event.key == pg.K_UP:
            self.runSingle(0,-1)
            #print('up')
          self.displayState()
    return self.score

if __name__ == "__main__":
  s = snakeHuman()
  s.run()
  print(s.score)