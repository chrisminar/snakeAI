"""Play a bunch of games and track stores."""

import numpy as np
from snake_reinforcement_learning import SnakeRL as snake

from helper import GRID_X, GRID_Y, NUM_SELF_PLAY_GAMES
from neural_net import NeuralNetwork
from play_games import PlayGames
from timer import Timer


# TODO merge this class with playgames
#####################
## self play class ##
#####################
class SelfPlay(PlayGames):
    """Generate self play games"""

    def play_games(self, generation: int, start_id: int, num_games: int = NUM_SELF_PLAY_GAMES):
        for i in range(num_games):
            with Timer() as t:
                g = snake(nn=self.neural_net,
                          sizeX=GRID_X, sizeY=GRID_Y)
                g.play(self.gamestate_to_nn)
                if len(g.moveList) > 1:
                    self.game_states.append(np.stack(g.stateList[:-1]))
                    self.heads.append(np.stack(g.headList[:-1]))
                    self.game_id.append(
                        np.full((len(g.stateList[:-1]), ), start_id+i))
                    self.prediction.append(np.array(g.moveList[:-1]))
                    self.scores.append(
                        np.full((len(g.stateList[:-1]), ), g.score))

        return self.gamestate_to_nn(np.concatenate(self.game_states)), np.concatenate(self.heads), np.concatenate(self.scores), np.concatenate(self.game_id), np.concatenate(self.prediction)
