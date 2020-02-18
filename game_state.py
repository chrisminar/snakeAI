import numpy as np
from globalVar import globe


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
