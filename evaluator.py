from globalVar import Globe as globe
from timer import Timer
from dataTrack import DataTrack
from neuralNet import NeuralNetwork
from snakeRL import snakeRL as snake
from playGames import PlayGames


#####################
## evaluator class ##
#####################
class Evaluator(PlayGames):
  """Evaluate a neural net"""

  def evaluate( self:Evaluator, generation:int, startID:int ):
    # play eval games
    with Timer() as overallTime:
      for i in range(globe.NUM_EVALUATION_GAMES):
        with Timer() as t:
          g = snake(nn=self.nn, sizeX=globe.GRID_X, sizeY=globe.GRID_Y)
          g.play()
          self.gamestates.append( np.stack(g.stateList))
          self.gameScore.append(  np.full( (len(g.stateList), ), g.score ) )
          self.gameID.append(     np.full( (len(g.stateList), ), startID+i ) )
          self.prediction.append( np.array(g.moveList))
        self.dfTrack.appendEvaluatorDetail( t.secs, g.score, generation, startID+i, len(g.stateList), 0.0, 0.0, 0.0, 0.0) #todo, get the mcts times tracked
    self.dfTrack.appendEvaluatorBroad(overallTime.secs, self.dfTrack.evaluator_detail.loc[generation,'score'].mean())

    return self.gamestate_to_nn(np.concatenate(self.gamestates)), np.concatenate(self.gameScore), np.concatenate(self.gameID), np.concatenate(self.prediction)