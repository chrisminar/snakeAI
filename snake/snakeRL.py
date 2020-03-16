from snake import Snake
import random
import numpy as np
import pygame as pg

class SnakeRL(Snake):
  def __init__(self, *args, **kwargs):
    super(SnakeRL, self).__init__(kwargs['sizeX'], kwargs['sizeY']) #expect sizeX, sizeY
    self.nn = kwargs['nn']
    self.foodX, self.foodY = self.spawn_food()
    self.grid[self.foodX][self.foodY] = -2   #set food on grid
    self.stateList = []
    self.moveList = []

  def runStep(self, dir: str): 
    if not self.gameover:
      self.moves+=1
      if dir[3]: #left
        self.runSingle(-1, 0)
      elif dir[1]: #right
        self.runSingle(1, 0)
      elif dir[2]: #down
        self.runSingle(0, 1)
      elif dir[0]: #up
        self.runSingle(0, -1)
      else: #invalid direction = no input
        self.runSingle(self.Xdir, self.Ydir)
      self.stateList.append(self.grid)
    return

  def play(self):
    while (self.gameover == False):
      newDir = self.evaluateNextStep()
      self.moveList.append(newDir)
      self.runStep(newDir)
    return self.score

  def evaluateNextStep(self):
    policy, value = self.nn.evaluate(self.grid)
    return np.argmax(policy)