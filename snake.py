###snake

import random
import numpy as np
import pygame as pg
buttons = {0: "up",
           1: "right",
           2: "down",
           3: "left",
           4: "none"}

class snake():
  def __init__(self, isHuman = True, sizeX=20, sizeY=20):
    #grid
    self.sizeX = sizeX                       #width of grid
    self.sizeY = sizeY                       #height of grid
    self.grid = np.zeros((self.sizeX,self.sizeY))-1 #grid                  -2 = food, -1 = empty, any number <= 0 is the position of the snakes body. eg if the snake has length 10 then 0 is the head and 9 is the tail

    #snake
    self.X = int(self.sizeX/2)               #snake head position x
    self.Y = int(self.sizeY/2)               #snake head position y
    self.length = 2                          #current length of snake
    self.grid[self.X][self.Y] = 0            #set snake head on the grid

    #scoring
    self.score = 0                           #current score
    self.score_per_food = 100                #point modification for eating food
    self.score_per_move = -1                 #point modificaiton for moving
    self.gameoverThreshold = -600

    currentstate = random.getstate()         #set random seed without ruining the other seed
    random.seed(0)
    self.randomState = random.getstate()
    random.setstate(currentstate)

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
    self.isHuman = isHuman
    if self.isHuman:
      self.humanGameInit()
    else:
      self.aiGameInit()

  def humanGameInit(self):
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

  def humanRun(self):
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

  def aiRunStep(self, dir: str): 
    if not self.gameover:
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
      self.score += (self.score_per_food*self.length)
      ateThisTurn = True
      if self.length > 398: #if snake is max length, the game has been won
        self.score+= 10000
        self.gameover = True

    #move body
    for i in range(self.sizeX):
      for j in range(self.sizeY):
        if self.grid[i][j] >= 0:
          self.grid[i][j] += 1
          if (self.grid[i][j] == self.length-1) and (not ateThisTurn): # if the gird length is longer than the actual length, delete the tail
            self.grid[i][j] = -1

    #check if dead
    self.gameover = self.check_game_over()

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
    currentState = random.getstate() #save random state
    random.setstate(self.randomState) #go to semi-random state
    if count > 1:
      spot = random.randint(1,count-1)
    else:
      print('no more valid spots')
      return(i,j)
    self.randomState = random.getstate() #save semi random-state
    random.setstate(currentState)#go back to random state

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
    if (self.X < 0) or (self.X >= self.sizeX) or self.score < self.gameoverThreshold:
      return True
    elif (self.Y < 0) or (self.Y >= self.sizeY):
      return True

    #check if we ran into the body
    if (self.grid[self.X][self.Y] >= 0):
      return True

    return False

class snake_test():
  def __init__(self):
    self.snake = snake(8,8)
    self.count = 0
    self.passed = 0
    self.failed = 0

  def run_tests(self):
    self.test_spawn_food()
    self.test_check_game_over()
    print('passed',self.passed)
    print('failed',self.failed)

  def test_spawn_food(self):
    print('testing spawn food')
    self.snake.grid = np.zeros((8,8))
    for k in range(0,7):
      self.snake.length = 0
      #make grid
      for i in range(8):
        for j in range(8):
          if (i == k) or (i==k+1):
            self.snake.grid[i][j] = self.snake.length
            self.snake.length+=1
          else:
            self.snake.grid[i][j] = -1
      #test
      (x,y) = self.snake.spawn_food()
      self.snake.grid[x][y] = -2
      if (x == k) or (x==k+1):
        print(self.snake.grid)
        print('food location',x,y)
        print (self.count,'failed')
        self.failed += 1
      else:
        #print (self.count,'passed')
        self.passed += 1
      self.count += 1
  
  def test_check_game_over(self):
    print('testing check game over')
    self.snake.length = 0
    #set grid
    for i in range(8):
        for j in range(8):
          if (i == 4) or (i == 5):
            self.snake.grid[i][j] = self.snake.length
            self.snake.length+=1
          else:
            self.snake.grid[i][j] = -1
    #sweep through possibilities
    test = False
    for i in range(-1,10):
      for j in range(-1,10):
        self.snake.X = i
        self.snake.Y = j
        self.snake.gameover = False
        test = self.snake.check_game_over()
        #check out of bounds
        if (i == -1) or (i == 9) or (j == -1) or (j == 9):
          if test == True:
            #print (self.count,'passed')
            self.passed += 1
          else:
            print (self.count,'failed')
            self.failed += 1
        #check for overlap with snake
        elif (i == 4) or (i == 5):
          if test == True:
            #print (self.count,'passed')
            self.passed += 1
          else:
            print (self.count,'failed')
            self.failed += 1
        else:
          if test == False:
            #print (self.count,'passed')
            self.passed += 1
          else:
            print (self.count,'failed')
            self.failed += 1
        self.count += 1

if __name__ == "__main__":
  #s = snake_test()
  #s.run_tests()
  s = snake()
  s.run()
  print(s.score)