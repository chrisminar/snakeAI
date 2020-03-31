import unittest
from trainSnakeRL import TrainRL
from neuralNet import NeuralNetwork
from selfPlay import SelfPlay
from dataTrack import DataTrack
import numpy as np

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
    self.assertEqual(0,1)

if __name__ == '__main__':
  unittest.main()


