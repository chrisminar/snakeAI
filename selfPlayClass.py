#todo, convert and store gamelist as nn state
from globalVar import globe
from game_state import game_state
from game_state import game
from timer import Timer
from dataTrack import dataTrack
from neuralNet import neural_network


#####################
## self play class ##
#####################
class selfPlayClass():
  """description of class"""
  def __init__( self, dfTrack:dataTrack, nn:neural_network ):
    self.gamelist = []
    self.dfDetail = dfDetail
    self.nn = nn


  def playGames( self, generation ):
    for i in range(globe.NUM_TRAINING_GAMES):
      with Timer as t():
        g = game(self.nn)
        gamelist.append(game.playgame())
      self.dfTrack.appendSelfPlayDetail(t.secs, gamelist[-1].snakeInstance.score, generation, i)
    self.dfTrack.appendSelfPlayBroad(self.dfTrack.self_play_detail.loc[generation,'time'].sum(), self.dfTrack.self_play_detail.loc[generation,'score'].mean())