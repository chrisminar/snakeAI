import unittest
from dataTrack import DataTrack

class DataTrack_test(unittest.TestCase):

  #test self play detail
  def test_add_to_self_play_detail(self):
    dt = DataTrack()
    l1 = len(dt.self_play_detail.index.values)
    val = [0.0, 1, 2]
    dt.appendSelfPlayDetail(val[0], val[1], 0, val[2])
    self.assertGreater(len(dt.self_play_detail.index.values) , l1, "failed to add new row to self play detail dataframe")
    err = dt.self_play_detail_column_names
    for v, col in zip(val, err):
      self.assertEqual(dt.self_play_detail.loc[0][col] , v, "failed to set {} for self play detail dataframe".format(col))

  #test self play broad
  def test_add_to_self_play_broad(self):
    dt = DataTrack()
    l1 = len(dt.self_play_broad.index.values)
    val = [0.0, 1.0]
    dt.appendSelfPlayBroad(val[0], val[1])
    self.assertGreater(len(dt.self_play_broad.index.values) , l1, "failed to add new row to self play broad dataframe")
    err = dt.self_play_broad_column_names
    for v, col in zip(val, err):
      self.assertEqual(dt.self_play_broad.loc[0][col] , v, "failed to set {} for self play broad dataframe".format(col))

  #test evaluator detail
  def test_add_to_evaluator_detail(self):
    dt = DataTrack()
    l1 = len(dt.evaluator_detail.index.values)
    val = [0.0, 1, 3, 4, 5.0, 6.0, 7.0, 8.0]
    dt.appendEvaluatorDetail(val[0], val[1], 0, val[2], val[3], val[4], val[5], val[6], val[7])
    self.assertGreater(len(dt.evaluator_detail.index.values) , l1, "failed to add new row to evaluator detail dataframe")
    err = dt.evaluator_detail_column_names
    for v, col in zip(val, err):
      self.assertEqual(dt.evaluator_detail.loc[0][col] , v, "failed to set {} for evaluator detail dataframe".format(col))

  # test evaluator broad
  def test_add_to_evaluator_broad(self):
    dt = DataTrack()
    l1 = len(dt.evaluator_broad.index.values)
    val = [0.0, 1.0]
    dt.appendEvaluatorBroad(val[0], val[1])
    self.assertGreater(len(dt.evaluator_broad.index.values) , l1, "failed to add new row to evaluator broad dataframe")
    err = dt.evaluator_broad_column_names
    for v, col in zip(val, err):
      self.assertEqual(dt.evaluator_broad.loc[0][col] , v, "failed to set {} for evaluator detail dataframe".format(col))

  #test training
  def test_add_to_trainer(self):
    dt = DataTrack()
    l1 = len(dt.training.index.values)
    val = [0.0, 1, 2.0]
    dt.appendTraining(val[0], val[1], val[2])
    self.assertGreater(len(dt.training.index.values) , l1, "failed to add new row to training dataframe")
    err = dt.training_column_names
    for v, col in zip(val, err):
      self.assertEqual(dt.training.loc[0][col] , v, "failed to set {} for training dataframe".format(col))

if __name__ == '__main__':
  unittest.main()
