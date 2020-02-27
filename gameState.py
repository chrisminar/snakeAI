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


class Game:
  def __init__(self, nn:NeuralNetwork):
    self.snakeInstance = snake(False, globe.GRID_X,globe.GRID_Y)
    self.gamestate_to_nn = np.vectorize(game.grid_val_to_nn)
    return

  def grid_val_to_nn(input):
    if input == -1:
      out = 1
    elif input >0:
      out = -1
    elif input == 0:
      out = -2
    else:
      out = 2

  def concatenate_game_output(self, snakeInstance:snake):
    game_states = np.stack( snakeInstance.states ) #todo need to have a list of game states
    scores = np.full( ( game_states.shape[0], ), snakeInstance.score )
    return [self.gamestate_to_nn(game_states), scores]

  def playgame(self):
    self.snakeInstance.play()
    return concatenate_game_output(self.snakeInstance)
