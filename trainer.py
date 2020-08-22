from globalVar import Globe as globe
from timer import Timer
from neuralNet import NeuralNetwork
import numpy as np


###################
## trainer class ##
###################
class Trainer():
  """description of class"""
  def __init__( self, nn:NeuralNetwork ):
    self.nn = nn

  def train( self, generation:int, inputs, heads, move_predictions):
    with Timer() as t:
      statesP = np.copy(inputs) #copy input grids for reshaping
      statesP = np.reshape(statesP, (statesP.shape[0], statesP.shape[1], statesP.shape[1], 1)) #reshape

      #train on them
      self.nn.train( statesP, heads, move_predictions, generation )
      num_minibatch = statesP.shape[0]/globe.BATCH_SIZE