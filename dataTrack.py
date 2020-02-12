

class dataTrack:
  def __init__(self):
    self.self_play_detail_column_names = ['time', 'score',      'generation', 'game_id']
    self.self_play_broad_column_names  = ['time', 'mean score']
    self.evaluator_detail_column_names = ['time', 'score',      'generation', 'game_id', 'game_length', 'mcts_tree_total_time', 'mcts_nn_total_time', 'mcts_tree_mean_time', 'mcts_nn_mean_time']
    self.evaluator_broad_column_names  = ['time', 'mean_score']
    self.training_column_names         = ['time', 'num_minibatch', 'mean_time_per_minibatch']
    self.self_play_detail = pd.DataFrame(columns=self.self_play_detail_column_names) #index is #
    self.self_play_broad  = pd.DataFrame(columns=self.self_play_broad_column_names) #index is generation #
    self.evaluator_detail = pd.DataFrame(columns=self.evaluator_detail_column_names) #index is #
    self.evaluator_broad  = pd.DataFrame(columns=self.evaluator_broad_column_names) #index is #
    self.training         = pd.DataFrame(columns=self.training_column_names) #index is generation #

  def appendSelfPlayDetail(self, time:float, score:int, generation:int, game_id:int):
    current_index = len(self.self_play_detail.index.values)
    self.self_play_detail.loc[current_index] = [time, score, generation, game_id]

  def appendSelfPlayBroad(self, time:float, mean_score:int):
    current_index = len(self.self_play_broad.index.values)
    self.self_play_broad.loc[current_index] = [time, mean_score]

  def appendEvaluatorDetail(self, time:float, score:int, generation:int, game_id:int, game_length:int, mcts_tree_total_time:float, mcts_nn_total_time:float, mcts_tree_mean_time:float, mcts_nn_mean_time:float):
    current_index = len(self.evaluator_detail.index.values)
    self.evaluator_detail.loc[current_index] = [time, score, generation, game_id, game_length, mcts_tree_total_time, mcts_nn_total_time, mcts_tree_mean_time, mcts_nn_mean_time]

  def appendEvaluatorBroad(self, time:float, score:int):
    current_index = len(self.evaluator_broad.index.values)
    self.evaluator_broad.loc[current_index] = [time, score]

  def appendTraining(self, time:float, num_minibatch:int, mean_time_per_minibatch:float):
    current_index = len(self.training.index.values)
    self.training.loc[current_index] = [time, num_minibatch, mean_time_per_minibatch]
