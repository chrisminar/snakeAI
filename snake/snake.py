###snake

import random
import numpy as np
import pygame as pg
from globalVar import Globe as globe

buttons = {0: "up",
           1: "right",
           2: "down",
           3: "left",
           4: "none"}

class Snake():
  def __init__(self, sizeX=20, sizeY=20):
    #grid
    self.sizeX = sizeX                       #width of grid
    self.sizeY = sizeY                       #height of grid
    self.grid = np.zeros((self.sizeX,self.sizeY))-1 #grid                  -2 = food, -1 = empty, any number <= 0 is the position of the snakes body. eg if the snake has length 10 then 0 is the head and 9 is the tail

    #snake
    self.X = int(0)               #snake head position x
    self.Y = int(0)               #snake head position y
    self.length = 0                          #current length of snake
    self.grid[self.X][self.Y] = 0            #set snake head on the grid

    #scoring
    self.score = 0                           #current score
    self.score_per_food = 100                #point modification for eating food
    self.score_per_move = 0                  #point modificaiton for moving
    self.score_penalty_for_failure = -50     #point modification for dying
    self.gameoverThreshold = -self.sizeX*self.sizeY*2
    self.moveThreshold = self.sizeX*self.sizeY*2
    self.moves = 0

    self.movesSinceFood = 0

    #input
    dir = random.randint(0,3)
    if dir == 0:
      self.Xdir = 0                           #0,1,2,3 = up,right,down,left
      self.Ydir = -1
    elif dir == 1:
      self.Xdir = 1
      self.Ydir = 0
    elif dir == 2:
      self.Xdir = 0
      self.Ydir = -1
    else:
      self.Xdir = -1
      self.Ydir = 0

    #gamestate
    self.gameover = False

    self.foodX,self.foodY = self.spawn_food()

  def runSingle(self, xdir, ydir):
    self.Xdir = xdir
    self.Ydir = ydir
    self.step_time()

  def displayState(self):
    #draw grid
    #font = pg.font.Font('freesansbold.ttf',12)
    for i in range(self.sizeX):
      for j in range(self.sizeY):
        pg.draw.rect(self.DISPLAY, self.gridNum2Color(self.grid[i][j]), (i*21, j*21, 20, 20))
        #text = font.render(str(int(self.grid[i][j])), True, (255,255,255))
        #textRect = text.get_rect()
        #textRect.center = (i*21+10,j*21+10)
        #self.DISPLAY.blit(text,textRect)
    pg.display.update()

  def gridNum2Color(self, num):
    if num == -2:#food
      return (255,0,0)
    elif num == 0:#head
      return (0,0,0)
    elif num >0:#tail
      return (100,100,100)
    else:#background
      return (200, 200, 200)

  def step_time(self):
    #move head
    self.X += self.Xdir
    self.Y += self.Ydir
    self.score += self.score_per_move

    #check if snake ate
    ateThisTurn = False;
    if (self.X == self.foodX) and (self.Y == self.foodY):
      self.length += 1
      self.foodX,self.foodY = self.spawn_food()
      self.grid[self.foodX][self.foodY] = -2
      self.score += (self.score_per_food)
      ateThisTurn = True
      self.moves = 0
      if self.length > 15: #if snake is max length, the game has been won
        self.score+= 10000
        self.gameover = True

    #move body
    for i in range(self.sizeX):
      for j in range(self.sizeY):
        if self.grid[i][j] >= 0:
          self.grid[i][j] += 1
          if (self.grid[i][j] > self.length) and (not ateThisTurn): # if the gird length is longer than the actual length, delete the tail
            self.grid[i][j] = -1

    #check if dead
    self.gameover = self.check_game_over()
    if self.gameover:
      self.score += self.score_penalty_for_failure

    if (self.gameover == False):
      #set head on grid
      self.grid[self.X][self.Y] = 0

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

    #print(mask)
    #print(spot, count)

    # find the x and y location of the spot
    for i in range(self.sizeX):
      for j in range(self.sizeY):
        if (mask[i][j] == spot):
          return (i,j)
    print('not found')
    return (i,j)

  def check_game_over(self):
    #check if we ran into a wall
    if (self.X < 0) or (self.X >= self.sizeX) or self.score < self.gameoverThreshold or self.moves > self.moveThreshold:
      return True
    elif (self.Y < 0) or (self.Y >= self.sizeY):
      return True
    elif self.movesSinceFood > globe.TIMEOUT:
      return True


    #check if we ran into the body
    if (self.grid[self.X][self.Y] >= 0):
      return True

    return False