from snake import Snake
import random
import numpy as np
import pygame as pg
from globalVar import Globe as globe
from timer import Timer

class SnakeRL(Snake):
  def __init__(self, *args, **kwargs):
    super(SnakeRL, self).__init__(kwargs['sizeX'], kwargs['sizeY']) #expect sizeX, sizeY
    self.nn = kwargs['nn']
    self.foodX, self.foodY = self.spawn_food()
    self.grid[self.foodX][self.foodY] = -2   #set food on grid
    self.stateList = []
    self.moveList = []
    self.headList = []

  def runStep(self, dir: str):
    if not self.gameover:
      self.moves += 1
      self.movesSinceFood += 1
      if dir==3: #left
        self.runSingle(-1, 0)
      elif dir==1: #right
        self.runSingle(1, 0)
      elif dir==2: #down
        self.runSingle(0, -1)
      elif dir==0: #up
        self.runSingle(0, 1)
      else: #invalid direction = no input
        self.runSingle(self.Xdir, self.Ydir)
    return

  def play(self, grid_fn):
    while (self.gameover == False):
      newDir, move, headView = self.evaluateNextStep(grid_fn)
      self.moveList.append(move)
      self.stateList.append(np.copy(self.grid))
      self.headList.append(headView)
      self.runStep(newDir)
    return self.score

  def evaluateNextStep(self, grid_fn):
    pre_processed_grid = grid_fn(self.grid) # preprocess grid
    headView = SnakeRL.convertHead(self.X, self.Y, self.sizeX, self.sizeY, self.grid)
    policy = self.nn.evaluate(pre_processed_grid, headView)
    
    out = [0, 0, 0, 0]
    newDir = np.argmax(policy).astype(int)

    # check if nn direction is dead
    if headView[newDir] == 0: #current trajotory is death
      if np.sum(headView) > 0: #at least one direction is free
        validIndex = np.where(headView)[0]
        newDir = np.random.choice(validIndex)
      else:
        pass#no directions are free, death is immenent
    else:
      pass#current trajectory is ok

    out[newDir] = 1

    return newDir, out, headView

  # look at head, return a boolean array [up, right, down, left] 0 means not ok to move, and 1 means ok to move
  def convertHead(x: int, y: int, sizeX: int, sizeY: int, grid: np.ndarray) -> np.ndarray:
    isFree = np.zeros((4,))

    if x == 0: #on left wall
      pass
    elif grid[x-1, y] < 0: #is left empty or food
      isFree[3] = 1

    if x == sizeX-1: #on right wall wall
      pass
    elif grid[x+1, y] < 0: #is right empty or food
      isFree[1] = 1

    if y == 0: #on bot wall
      pass
    elif grid[x, y-1] < 0: #is below empty or food
      isFree[2] = 1

    if y == sizeY-1: #on top wall
      pass
    elif grid[x, y+1] < 0: #is above empty or food
      isFree[0] = 1

    return isFree