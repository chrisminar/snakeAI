#todo convert gamestates to training
  #figure out what the training input format needs to be
import numpy as np
from globalVar import Globe as globe


class GameState:
  #game state
  # array gridx x gridy int[][]
  # food 2
  # empty 1
  # body -1
  # head -2
  def __init__(self, grid=np.zeros(globe.GRID_X,globe.GRID_Y)):
    self.x = globe.GRID_X
    self.y = globe.GRID_Y
    self.grid = grid

  def move(grid, policy, score:int):
    """Move the snake in a given direction"""
    direction = np.argmax(policy)
    if direction == 0: # 0,1,2,3 = up,right,down,left
      xd = 0
      yd = -1
    elif direction == 1:
      xd = 1
      yd = 0
    elif direction == 2:
      xd = 0
      yd = -1
    else:
      xd = -1
      yd = 0

    #head position
    head = np.argwhere(grid==0)
    x = head[0]
    y = head[1]

    #move head
    x += xd
    y += yd

    #food position
    food = np.argwhere(grid==2)
    xf = food[0]
    xy = food[1]

    minV = np.min(grid)
    tailPosition = np.unravel_index(np.argmin(grid, axis=None), grid.shape)
    xt = tailPosition[0]
    yt = tailPosition[0]

    length = np.sum(grid,axis=(0,1))

    #check if snake ate
    ateThisTurn = False;
    if ( x == xf ) and ( y == yf ):
      length += 1
      fx,fy = GameState.spawn_food()
      grid[fx][fy] = 2 #set new food
      score += ( globe.SCORE_PER_FOOD * length )
      ateThisTurn = True

    #move body
    for i in range(globe.GRID_X):
      for j in range(globe.GRID_Y):
        if grid[i][j] < 1:
          grid[i][j] -= 1
    if not ateThisTurn: # if snake didn't eat, set tail to empty
      grid[xt][yt] = 1

    go = GameState.check_game_over(x,x,grid)
    if not go:
      grid[x][y] = 0

    return grid, score, direction, go

  def check_game_over(x,y,grid):
    #check if we ran into a wall
    if (x < 0) or (y >= globe.GRID_X):
      return True
    elif (y < 0) or (y >= globe.GRID_Y):
      return True

    #check if we ran into the body
    if (grid[x][y] <= 0):
      return True

    return False

  def spawn_food()