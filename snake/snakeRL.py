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

  def spawn_food(self):
    #generate mask matrix and count empty spots
    mask = np.zeros((self.sizeX,self.sizeY))
    count = 1
    for i in range(self.sizeX):
      for j in range(self.sizeY):
        if self.grid[i][j] == -1:
          mask[i][j] = count
          count += 1

    #generate a random number from 0-count
    if count > 1:
      spot = random.randint(1,count-1)
    else:
      print('no more valid spots')
      return(i,j)

    # find the x and y location of the spot
    for i in range(self.sizeX):
      for j in range(self.sizeY):
        if (mask[i][j] == spot):
          return (i,j)
    print('not found')
    return (i,j)

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