import unittest

import numpy as np
from helper import Globe as globe
from neural_net import NeuralNetwork
from self_play import SelfPlay
from train_snake_reinforcement_learning import TrainRL


class trainSnake_test(unittest.TestCase):
    def test_init(self):
        t = TrainRL()

    def test_addGamesToList(self):
        t = TrainRL()
        nn = NeuralNetwork()
        spc = SelfPlay(nn)
        states, heads, scores, ids, moves = spc.play_games(0, 0, 5)
        t.add_games_to_list(states, heads, scores, ids, moves)
        self.assertTrue(np.max(t.game_ids), 4)
        self.assertEqual(
            t.game_states.shape[0], t.game_ids.shape[0], 'bad gameid shape')
        self.assertEqual(
            t.game_states.shape[0], t.game_scores.shape[0], 'bad score shape')
        self.assertEqual(
            t.game_states.shape[0], t.moves.shape[0], 'bad moves shape')

    def test_trimGameList(self):
        num_games = NUM_TRAINING_GAMES
        moves_per_game = 3
        n = num_games*moves_per_game
        moves = np.zeros((n, 4))
        states = np.zeros((n, GRID_X, GRID_Y))
        scores = np.zeros((n,))
        gameIDs = np.zeros((moves_per_game,))
        heads = np.ones((n, 4))
        for i in range(1, num_games):
            gameIDs = np.concatenate(
                [gameIDs, np.zeros((moves_per_game,)) + i])
        t = TrainRL()
        t.add_games_to_list(states, heads, scores, gameIDs, moves)

        offset = gameIDs[-1]+1
        gameIDs = np.zeros((moves_per_game,)) + offset
        for i in range(1, num_games):
            gameIDs = np.concatenate(
                [gameIDs, np.zeros(moves_per_game,) + i + offset])
        t.add_games_to_list(states, heads, scores+100, gameIDs, moves)
        t.trim_game_list()
        self.assertEqual(
            t.game_states.shape[0], NUM_TRAINING_GAMES*moves_per_game, 'Gamestates trim unsuccessful')
        self.assertEqual(
            t.moves.shape[0], NUM_TRAINING_GAMES*moves_per_game, 'Moves trim unsuccessful')
        self.assertEqual(
            t.game_scores.shape[0], NUM_TRAINING_GAMES*moves_per_game, 'Scores trim unsuccessful')
        self.assertEqual(
            t.game_ids.shape[0], NUM_TRAINING_GAMES*moves_per_game, 'game ids trim unsuccessful')
        self.assertEqual(np.min(t.game_scores), 100, 'min scores not removed')


if __name__ == '__main__':
    unittest.main()