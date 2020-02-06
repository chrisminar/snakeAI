
import snakeRL


def testDataTrack():
  dt = snakeRL.dataTrack()

  #test self play detail
  l1 = len(dt.self_play_detail.index.values)
  val = [0.0, 1, 2, 3]
  dt.appendSelfPlayDetail(val[0], val[1], val[2], val[3])
  assert len(dt.self_play_detail.index.values) > l1, "failed to add new row to self play detail dataframe"
  err = dt.self_play_detail_column_names
  for v, col in zip(val, err):
    assert dt.self_play_detail.loc[0][col] == v, "failed to set {} for self play detail dataframe".format(col)

  #test self play broad
  l1 = len(dt.self_play_broad.index.values)
  val = [0.0, 1]
  dt.appendSelfPlayBroad(val[0], val[1])
  assert len(dt.self_play_broad.index.values) > l1, "failed to add new row to self play broad dataframe"
  err = dt.self_play_broad_column_names
  for v, col in zip(val, err):
    assert dt.self_play_broad.loc[0][col] == v, "failed to set {} for self play broad dataframe".format(col)

  #test evaluator detail
  l1 = len(dt.evaluator_detail.index.values)
  val = [0.0, 1, 2, 3, 4, 5.0, 6.0, 7.0, 8.0]
  dt.appendEvaluatorDetail(val[0], val[1])
  assert len(dt.evaluator_detail.index.values) > l1, "failed to add new row to evaluator detail dataframe"
  err = dt.evaluator_detail_column_names
  for v, col in zip(val, err):
    assert dt.evaluator_detail.loc[0][col] == v, "failed to set {} for evaluator detail dataframe".format(col)

  # test evaluator broad
  l1 = len(dt.evaluator_broad.index.values)
  val = [0.0, 1]
  dt.appendEvaluatorBroad(val[0], val[1])
  assert len(dt.evaluator_broad.index.values) > l1, "failed to add new row to evaluator broad dataframe"
  err = dt.evaluator_broad_column_names
  for v, col in zip(val, err):
    assert dt.evaluator_broad.loc[0][col] == v, "failed to set {} for evaluator detail dataframe".format(col)

  #test training
  l1 = len(dt.training.index.values)
  val = [0.0, 1, 2.0]
  dt.appendTraining(val[0], val[1])
  assert len(dt.training.index.values) > l1, "failed to add new row to training dataframe"
  err = dt.training_column_names
  for v, col in zip(val, err):
    assert dt.training.loc[0][col] == v, "failed to set {} for training dataframe".format(col)