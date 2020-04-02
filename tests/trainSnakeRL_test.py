import unittest
from trainSnakeRL import TrainRL
from neuralNet import NeuralNetwork
from selfPlay import SelfPlay
from dataTrack import DataTrack
import numpy as np
from globalVar import Globe as globe

class trainSnake_test(unittest.TestCase):
  def test_init(self):
    t = TrainRL()

  def test_addGamesToList(self):
    t=TrainRL()
    tr = DataTrack()
    nn=NeuralNetwork()
    spc = SelfPlay(tr, nn)
    states, scores, ids, moves = spc.playGames(0, 0, 5)
    t.addGamesToList(states,scores,ids,moves)
    self.assertTrue(np.max(t.gameIDs), 4)
    self.assertEqual(t.gameStates.shape[0], t.gameIDs.shape[0], 'bad gameid shape')
    self.assertEqual(t.gameStates.shape[0], t.gameScores.shape[0], 'bad score shape')
    self.assertEqual(t.gameStates.shape[0], t.moves.shape[0], 'bad moves shape')

  def test_trimGameList(self):
    num_games = globe.NUM_TRAINING_GAMES
    moves_per_game = 3
    n=num_games*moves_per_game
    moves = np.zeros((n, 4))
    states = np.zeros((n,10,10))
    scores = np.zeros((n,))
    gameIDs = np.zeros((moves_per_game,))
    for i in range(1,num_games):
      gameIDs = np.concatenate([gameIDs, np.zeros((moves_per_game,)) + i ])
    t = TrainRL()
    t.addGamesToList(states,scores,gameIDs,moves)

    offset = gameIDs[-1]+1
    gameIDs = np.zeros((moves_per_game,))+offset
    for i in range(1,num_games):
      gameIDs = np.concatenate([gameIDs, np.zeros(moves_per_game,) + i + offset])
    t.addGamesToList(states,scores+100,gameIDs,moves)
    t.trimGameList()
    self.assertEqual(t.gameStates.shape[0], globe.NUM_TRAINING_GAMES*moves_per_game, 'Gamestates trim unsuccessful')
    self.assertEqual(t.moves.shape[0], globe.NUM_TRAINING_GAMES*moves_per_game, 'Moves trim unsuccessful')
    self.assertEqual(t.gameScores.shape[0], globe.NUM_TRAINING_GAMES*moves_per_game, 'Scores trim unsuccessful')
    self.assertEqual(t.gameIDs.shape[0], globe.NUM_TRAINING_GAMES*moves_per_game, 'game ids trim unsuccessful')
    self.assertEqual(np.min(t.gameScores),100, 'min scores not removed')

if __name__ == '__main__':
  unittest.main()


