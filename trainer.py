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

  def train( self, generation:int, inputs, scores, move_predictions, meanScore:float):
    with Timer() as t:
      #take top x scores from those games
      idx = np.argwhere(scores>meanScore)

      #get all permutations
      statesP, movesP, scoresP = Trainer.permute_inputs(inputs, scores, move_predictions, idx)
      
      #train on them
      self.nn.train( statesP, scoresP, movesP, generation )
      num_minibatch = mp.shape[0]
    self.dfTrack.appendTraining( t.secs, num_minibatch, t.secs/num_minibatch )

  def permute_inputs(states, scores, moves, idx):
    #flip left - right
    stateLR = np.flip(states[idx][:][:],1)
    movesLR = Trainer.flipMoveLR(moves, idx)
    scoreLR = np.copy(scores[idx])

    #flip ud
    stateUD = np.flipud(states[idx][:][:],2)
    movesUD = Trainer.flipMoveUD(moves,idx)
    scoreUD = np.copy(scores[idx])

    #rotate 90
    stateR90 = np.rot90(states[idx][:][:],1,(1,2))
    scoreR90 = np.copy(scores[idx])
    movesR90 = Trainer.rotateMoves(moves, idx, 1)

    stateR180 = np.rot90(states[idx][:][:],2,(1,2))
    scoreR180 = np.copy(scores[idx])
    movesR180 = Trainer.rotateMoves(moves, idx, 2)

    stateR270 = np.rot90(states[idx][:][:],3,(1,2))
    scoreR270 = np.copy(scores[idx])
    movesR270 = Trainer.rotateMoves(moves, idx, 3)

    stateOut = np.vstack(states[idx][:][:], stateLR, stateUD, stateR90, stateR180, stateR270)
    movesOut = np.vstack(moves[idx][:], movesLR, movesUD, movesR90, movesR180, movesR270)
    scoreOut = np.vstack(scores[idx], scoreLR, scoreUD, scoreR90, scoreR180, scoreR270)

    return stateOut, movesOut, scoreOut

  def flipMoveLR(moves, idx):
    movesLR = np.copy(moves[idx][:])
    movesLR[:][3], movesLR[:][1] = movesLR[:][1], movesLR[:][3]
    return movesLR

  def flipMoveUD(moves, idx):
    movesUD = np.copy(moves[idx][:])
    movesUD[:][0], movesUD[:][2] = movesUD[:][2], movesUD[:][0]
    return movesUD

  def rotateMoves(moves, idx, quads:int):
    return np.roll(moves[idx][:], -quads, axis=1)