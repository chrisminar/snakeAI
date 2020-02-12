
import numpy as np

class game_state:
  #game state
  # array gridx x gridy int[][]
  # food 2
  # empty 1
  # body -1
  # head -2
  def __init__(self):
    global GRID_X
    global GRID_Y
    self.x = GRID_X
    self.y = GRID_Y
    self.grid = np.zeros((self.x,self.y))
