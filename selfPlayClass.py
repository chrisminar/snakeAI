from globalVar import globe
from game_state import game_state
from game_state import game
from timer import Timer


#####################
## self play class ##
#####################
class selfPlayClass():
  """description of class"""
  def __init__(self, dfTrack):
    self.gamelist = []
    self.dfDetail = dfDetail

  def playGames(self, generation):
    for i in range(globe.NUM_TRAINING_GAMES):
      with Timer as t():
        gamelist.append(game.playgame())
      self.dfTrack.appendSelfPlayDetail(t.secs, gamelist[-1].snakeInstance.score, generation, i)
    self.dfTrack.appendSelfPlayBroad(self.dfTrack.self_play_detail.loc[generation,'time'].sum(), self.dfTrack.self_play_detail.loc[generation,'score'].mean())