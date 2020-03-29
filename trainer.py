from globalVar import Globe as globe
from gameState import GameState
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

  def train( self, generation:int, inputs, scores, move_predictions, meanScore:float):
    with Timer() as t:
      #take top x scores from those games
      idx = np.argwhere(scores>meanScore)

      #get all permutations
      statesP, movesP, scoresP = Trainer.permute_inputs(inputs[idx][:][:], scores[idx], move_predictions[idx][:])
      
      #train on them
      self.nn.train( statesP, scoresP, movesP, generation )
      num_minibatch = mp.shape[0]
    self.dfTrack.appendTraining( t.secs, num_minibatch, t.secs/num_minibatch )

  def permute_inputs(states, scores, moves):
    flipAxis = len(states.shape)-1
    #flip left - right
    stateLR = np.flip(states, axis=flipAxis-1)
    movesLR = Trainer.flipMoveLR(moves)
    scoreLR = np.copy(scores)

    #flip ud
    stateUD = np.flip(states, axis=flipAxis)
    movesUD = Trainer.flipMoveUD(moves)
    scoreUD = np.copy(scores)

    #rotate 90
    stateR90 = np.rot90(states,1,(1,2))
    scoreR90 = np.copy(scores)
    movesR90 = Trainer.rotateMoves(moves, 1)

    stateR180 = np.rot90(states,2,(1,2))
    scoreR180 = np.copy(scores)
    movesR180 = Trainer.rotateMoves(moves, 2)

    stateR270 = np.rot90(states,3,(1,2))
    scoreR270 = np.copy(scores)
    movesR270 = Trainer.rotateMoves(moves, 3)

    stateOut = np.vstack([states, stateLR, stateUD, stateR90, stateR180, stateR270])
    movesOut = np.vstack([moves, movesLR, movesUD, movesR90, movesR180, movesR270])
    scoreOut = np.concatenate([scores, scoreLR, scoreUD, scoreR90, scoreR180, scoreR270])

    return stateOut, movesOut, scoreOut

  def flipMoveLR(moves):
    movesLR = np.copy(moves)
    movesLR[:,3], movesLR[:,1] = moves[:,1], moves[:,3]
    return movesLR

  def flipMoveUD(moves):
    movesUD = np.copy(moves)
    movesUD[:,0], movesUD[:,2] = moves[:,2], moves[:,0]
    return movesUD

  def rotateMoves(moves, quads:int):
    return np.roll(moves, -quads, axis=1)