import numpy as np
from globalVar import globe
import snake

class game_state:
  #game state
  # array gridx x gridy int[][]
  # food 2
  # empty 1
  # body -1
  # head -2
  def __init__(self):
    self.x = globe.GRID_X
    self.y = globe.GRID_Y
    self.grid = np.zeros((self.x,self.y))


class game:
  def __init__(self):
    self.snakeInstance = snake.snake(False, globe.GRID_X,globe.GRID_Y)
    pass

  def gamestate_to_nn(game_state:game_state):
    pass

  def playgame(self):
    pass
