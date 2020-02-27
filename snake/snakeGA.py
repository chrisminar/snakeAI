from snake import Snake
import random
import numpy as np
import pygame as pg


buttons = {0: "up",
           1: "right",
           2: "down",
           3: "left",
           4: "none"}

class SnakeGA(Snake):
  def __init__(self, *args, **kwargs):
    super(snakeGA, self).__init__(*args, **kwargs)

    currentstate = random.getstate()         #set random seed without ruining the other seed
    random.seed(0)
    self.randomState = random.getstate()
    random.setstate(currentstate)
    self.foodX,self.foodY = self.spawn_food()
    self.grid[self.foodX][self.foodY] = -2   #set food on grid

  def aiRunStep(self, dir: str): 
    if not self.gameover:
      self.moves+=1
      if dir == buttons[3]: #left
        self.runSingle(-1, 0)
      elif dir == buttons[1]: #right
        self.runSingle(1,0)
      elif dir == buttons[2]: #down
        self.runSingle(0,1)
      elif dir == buttons[0]: #up
        self.runSingle(0,-1)
      elif dir == buttons[4]: #no input
        self.runSingle(self.Xdir,self.Ydir)
      else: #invalid direction = no input
        self.runSingle(self.Xdir,self.Ydir)
      return
    else: #game is over
      return