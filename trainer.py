from globalVar import Globe as globe
from gameState import GameState
from gameState import Game
from timer import Timer
from dataTrack import DataTrack
from neuralNet import NeuralNetwork
import numpy as np


###################
## trainer class ##
###################
class Trainer():
  """description of class"""
  def __init__( self, dfTrack:DataTrack, nn:NeuralNetwork ):
    self.td = trainingData
    self.dfTrack = dfTrack
    self.nn = nn

  def train( self, generation:int, inputs, scores, move_predictions ):
    with Timer() as t:
      inState = Trainer.randomizeMe(inputs)
      sc = np.vstack(scores,scores,scores,scores,scores,scores,scores)
      mp = np.vstack(move_predictions,move_predictions,move_predictions,move_predictions,move_predictions,move_predictions) #this is wrong
      self.nn.train( inState, sc, mp, generation )
      num_minibatch = mp.shape[0]
    self.dfTrack.appendTraining( t.secs, num_minibatch, t.secs/num_minibatch )

  def randomizeMe(states):
    stateLR = np.flip(states,1)
    stateUD = np.flipud(states,2)

    stateR90  = np.rot90(states,1,(1,2))
    stateR180 = np.rot90(states,2,(1,2))
    stateR270 = np.rot90(states,3,(1,2))

    return np.vstack(states, stateLR, stateUD, stateR90, stateR180, state270)