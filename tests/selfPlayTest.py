import unittest
from selfPlay import SelfPlay
from dataTrack import DataTrack
from neuralNet import NeuralNetwork
import numpy as np

class SelfPlay_test(unittest.TestCase):

  def test_PlayGames(self):
    
    dt = DataTrack()#make datatracker
    nn = NeuralNetwork()
    spc = SelfPlay(dt, nn)#make self play class
    #call play games
    state,head,score,id,prediction = spc.playGames(0, 0, num_games=2)

    print(dt.self_play_broad.head())
    print(dt.self_play_detail.head())
    print(state)
    print(score)
    print(id)
    print(prediction)
    #test datatracker braod and detail
    self.assertEqual(len(dt.self_play_broad.index.values), 1, 'Statistics not added to self play broad')
    self.assertEqual(len(dt.self_play_detail.index.values), 2, 'Statistics not added to self play detail')
    #test gamestate list
    self.assertGreater(state.shape[0],0)
    #test gamescore
    self.assertGreater(score.shape[0],0)
    #test gameid
    self.assertGreater(id.shape[0],0)
    self.assertEqual(np.max(id),1)
    #test prediction
    self.assertGreater(prediction.shape[0],0)
    self.assertLessEqual(np.max(prediction), 3)
    self.assertGreaterEqual(np.min(prediction), 0)
    #need to add a timeout function to snakerl

if __name__ == '__main__':
  unittest.main()

