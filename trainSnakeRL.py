#todo list
#change score tracking to be based on dataframe
#make generation the first input to all datatrack functions
#snake grid update can be done more efficiently

#goal -- snake runtime from 450s to 50s
#goal -- upgrade tensorflow to use gpu

#stuff to do someday
  # do network pruning
  # track/understand what the neural network is doing
  # gui
  # pause/play 
  #visualization of mcts

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import typing
from typing import List,Tuple
from dataTrack import DataTrack
from neuralNet import NeuralNetwork
from gameState import GameState
from globalVar import Globe as globe

from selfPlay import SelfPlay
from trainer import Trainer

class TrainRL():
  def __init__(self):
    self.tracker = DataTrack()
    self.gameStates = np.zeros((0,globe.GRID_X,globe.GRID_Y))
    self.gameScores = np.zeros((0,))
    self.gameIDs = np.zeros((0,))
    self.moves = np.zeros((0,4))
    self.nn = NeuralNetwork()
    self.gameID = 0
    return

  def train(self):
    generation = 0
    while 1:
      self.selfPlay(self.nn, generation)
      self.networkTrainer(generation)
      print('Mean score of {:0.1f} in generation {}. Selfplay, training in {}s, {}s'.format(self.tracker.self_play_broad.loc[generation,'mean_score'],
                                                                                            generation, 
                                                                                            self.tracker.self_play_broad.loc[generation, 'time'],
                                                                                            self.tracker.training.loc[generation, 'time']))
      generation += 1
    return

  #operates on:
    #neural network
  #outputs:
    #2000 game outputs
  def selfPlay(self, nn:NeuralNetwork, generation:int):
    spc = SelfPlay(self.tracker, nn)
    states, scores, ids, moves = spc.playGames(generation, self.gameID)
    self.gameID += globe.NUM_SELF_PLAY_GAMES
    self.addGamesToList(states, scores, ids, moves)
    self.trimGameList()
    return

  #operates on:
    # last 20000 games of self play
  #outputs:
    #new neural network
  def networkTrainer(self, generation:int):
    trn = Trainer(self.tracker, self.nn)
    trn.train(generation, self.gameStates, self.gameScores, self.moves, self.meanScore)
    return

  def addGamesToList(self, states, scores, ids, moves):
    self.gameStates = np.concatenate((self.gameStates, states))
    self.gameScores = np.concatenate((self.gameScores, scores))
    self.gameIDs    = np.concatenate((self.gameIDs, ids))
    self.moves      = np.concatenate((self.moves, moves))

  def trimGameList(self):
    minId = np.min(self.gameIDs)
    maxID = np.max(self.gameIDs)
    self.meanScore    = np.mean(self.gameScores)
    if (maxID - minId) > globe.NUM_TRAINING_GAMES:
      validIdx        = np.argwhere(self.gameIDs > (maxID - globe.NUM_TRAINING_GAMES) )
      self.gameIDs    = self.gameIDs[validIdx]
      self.gameScores = self.gameScores[validIdx]
      self.gameStates = self.gameStates[validIdx,:,:]
      self.moves      = self.moves[validIdx,:]
    return