#todo convert gamestates to training
  #figure out what the training input format needs to be
import numpy as np
from globalVar import Globe as globe
from snakeRL import SnakeRL as snake
from neuralNet import NeuralNetwork


class GameState:
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
