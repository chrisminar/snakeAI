import unittest

import numpy as np
from neuralNet import NeuralNetwork
from selfPlay import SelfPlay


class SelfPlay_test(unittest.TestCase):

    def test_play_games(self):

        nn = NeuralNetwork()
        spc = SelfPlay(nn)  # make self play class
        # call play games
        state, head, score, id, prediction = spc.playGames(0, 0, num_games=2)

        print(state)
        print(score)
        print(id)
        print(prediction)
        # test gamestate list
        self.assertGreater(state.shape[0], 0)
        # test gamescore
        self.assertGreater(score.shape[0], 0)
        # test gameid
        self.assertGreater(id.shape[0], 0)
        self.assertEqual(np.max(id), 1)
        # test prediction
        self.assertGreater(prediction.shape[0], 0)
        self.assertLessEqual(np.max(prediction), 3)
        self.assertGreaterEqual(np.min(prediction), 0)
        # need to add a timeout function to snakerl

    def test_grid_2_neural_network(self):
        nn = NeuralNetwork()
        spc = SelfPlay(nn)  # make self play class
        grid = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                grid[i, j] = i*4+j  # body/head
        grid[3, 3] = -2  # food
        grid[3, 2] = -1  # empty

        processed_grid = spc.gamestate_to_nn(grid)

        for i in range(4):
            for j in range(4):
                if grid[i, j] == -2:  # food
                    ans = -1
                elif grid[i, j] == -1:  # empty
                    ans = 0
                else:  # snake
                    ans = 1
                self.assertEqual(ans, processed_grid[i, j])


if __name__ == '__main__':
    unittest.main()
