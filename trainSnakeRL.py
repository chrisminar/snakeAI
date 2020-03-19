# self play
    #for the first 30 moves the temperature is set to 1, afterwards it is set to 0

# WORK BLOCKS
# ON DECK
  # UNIT TEST
    #do the actual work

#todo list
#the neural net being used isn't handled correctly in trainsnakerl
#make generation the first input to all datatrack functions
#change evaluator broad to explicity use the generation as the index
#change unit tests to reflect the class inheritance of selfplay/plagames/evaluator
#unit tests for evaluator
#change the training weights, drastically reduce the value of the 'value head' This doesn't really even need to be here because we are not using mcts
#move prediction vstack is wrong in trainer because I've flipped the grid
#change the 'evaluate nn' function to take (10,10) instead of (1,10,10,1)

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
from neuralNet import nn_out
from gameState import GameState
from globalVar import Globe as globe

from selfPlay import SelfPlay
from trainer import Trainer
from evaluator import Evaluator

class TrainRL():
  def __init__(self):
    self.tracker = DataTrack()
    self.gameStates = np.zeros((0,globe.GRID_X,globe.GRID_Y))
    self.gameScores = np.zeros((0,1))
    self.gameIDs = np.zeros((0,1))
    self.moves = np.zeros(0,4)
    self.currentNN = NeuralNetwork()
    self.bestNN = NeuralNetwork()
    self.bestNN_genID = 0
    self.gameID = 0
    return

  def train(self):
    generation = 0
    while 1:
      self.selfPlay(generation)
      self.networkTrainer(generation)
      self.evaluator(generation)
      print('Mean score of {:0.1f} in generation {}. Selfplay, training, evaluation in {}s, {}s, {}s'.format(self.tracker.evaluator_broad.loc[gneration, 'mean_score'],
                                                                                                             generation, 
                                                                                                             self.tracker.self_play_broad.loc[generation, 'time'],
                                                                                                             self.tracker.training.loc[generation, 'time'],
                                                                                                             self.tracker.evaluator_broad.loc[generation, 'time']))
      generation += 1
    return

  #operates on:
    #best neural network
  #outputs:
    #2000 game outputs
  def selfPlay(self, nn:NeuralNetwork, generation:int):
    spc = SelfPlay(self.tracker, self.bestNN)
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
    trn = Trainer(self.tracker)
    trn.train(generation, self.currentNN, self.gameStates, self.gameScores, self.moves)
    return

   #operates on: 
    #best neural network
    #current neural network
  #outputs:
    #400 games (with gamestate)
    #mean score of 400 games
  def evaluator(self, generation:int):
    eval = Evaluator(self.tracker, self.currentNN)
    states, scores, ids, moves = eval.evaluate(generation, self.gameID)
    if self.tracker.evaluator_broad.loc[generation, 'score'] == self.tracker.evaluator_broad['score'].max():
      self.bestNN_genID = generation
      self.bestNN.load(bestNN_genID)
      self.gameID += globe.NUM_EVALUATION_GAMES
      self.addGamesToList(states, scores, ids, moves)
      self.trimGameList()
    return

  def addGamesToList(self, states, scores, ids, moves):
    self.gameStates = np.concatenate(self.gameStates, states)
    self.gameScores = np.concatenate(self.gameScores, scores)
    self.gameIDs    = np.concatenate(self.gameIDs, ids)
    self.moves      = np.concatenate(self.moves, moves)

  def trimGameList(self):
    minId = np.min(self.gameIDs)
    maxID = np.max(self.gameIDs)
    if (maxID - minId) > globe.NUM_TRAINING_GAMES:
      validIdx        = np.argwhere(self.gameIDs > maxID - globe.NUM_TRAINING_GAMES)
      self.gameIDs    = self.gameIDs[validIdx]
      self.gameScores = self.gameScores[validIdx]
      self.gameStates = self.gameStates[validIdx,:,:]
    return