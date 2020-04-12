from snake import Snake
import random
import numpy as np
import pygame as pg
from globalVar import Globe as globe

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
      self.moves += 1
      self.movesSinceFood +=1
      if dir==3: #left
        self.runSingle(-1, 0)
      elif dir==1: #right
        self.runSingle(1, 0)
      elif dir==2: #down
        self.runSingle(0, 1)
      elif dir==0: #up
        self.runSingle(0, -1)
      else: #invalid direction = no input
        self.runSingle(self.Xdir, self.Ydir)
      
    return

  def play(self):
    while (self.gameover == False):
      newDir, move = self.evaluateNextStep()
      self.moveList.append(move)
      self.stateList.append(np.copy(self.grid))
      self.runStep(newDir)
    return self.score

  def evaluateNextStep(self):
    policy = self.nn.evaluate(self.grid)
    out = [0,0,0,0]
    out[int(np.argmax(policy))] = 1
    return np.argmax(policy), out