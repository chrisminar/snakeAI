from timer import Timer
from play_games import PlayGames
from neural_net import NeuralNetwork
from helper import Globe as globe
import numpy as np
from snake_reinforcement_learning import SnakeRL as snake


#####################
## self play class ##
#####################
class SelfPlay(PlayGames):
    """Generate self play games"""

    def play_games(self, generation: int, start_id: int, num_games: int = globe.NUM_SELF_PLAY_GAMES):
        for i in range(num_games):
            with Timer() as t:
                g = snake(nn=self.nn, sizeX=globe.GRID_X, sizeY=globe.GRID_Y)
                g.play(self.gamestate_to_nn)
                if len(g.moveList) > 1:
                    self.gamestates.append(np.stack(g.stateList[:-1]))
                    self.heads.append(np.stack(g.headList[:-1]))
                    self.gameId.append(
                        np.full((len(g.stateList[:-1]), ), start_id+i))
                    self.prediction.append(np.array(g.moveList[:-1]))
                    self.scores.append(
                        np.full((len(g.stateList[:-1]), ), g.score))

        return self.gamestate_to_nn(np.concatenate(self.gamestates)), np.concatenate(self.heads), np.concatenate(self.scores), np.concatenate(self.gameId), np.concatenate(self.prediction)
