from snake import Snake
import random
import numpy as np
import pygame as pg

class SnakeRL(Snake):
  def __init__(self, *args, **kwargs):
    super(snakeRL, self).__init__(*args, **kwargs)
    self.nn = kwargs['nn']
    self.foodX, self.foodY = self.spawn_food()
    self.grid[self.foodX][self.foodY] = -2   #set food on grid
    self.stateList = []

  def runStep(self, dir: str): 
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
      self.stateList.append(self.grid)
      return
    else: #game is over
      return

  def play(self):
    while (self.gameover == False):
      newDir = self.evaluateNextStep()
      self.runStep(newDir)
    return self.score

  def evaluateNextStep(self):
    #do some stuff
    direction = (0,0)
    return direction