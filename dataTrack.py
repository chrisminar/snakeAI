#todo write functions for save and load dataframes
#todo: game id should be a second index, not a column
#todo: move functionality that computes 'broad' statistics from 'detailed' statistics into a function in this class
import pandas as pd


class DataTrack:
  def __init__(self):
    index = ['generation','id']
    midx = pd.MultiIndex(levels = [[],[]], codes = [[],[]], names = index)
    self.self_play_detail_column_names = ['time', 'score', 'game_id']
    self.self_play_broad_column_names  = ['time', 'mean_score']
    self.training_column_names         = ['time', 'num_minibatch', 'mean_time_per_minibatch']
    self.self_play_detail = pd.DataFrame(columns=self.self_play_detail_column_names, index = midx) #index is gen #
    self.self_play_broad  = pd.DataFrame(columns=self.self_play_broad_column_names) #index is generation #
    self.training         = pd.DataFrame(columns=self.training_column_names) #index is generation #

  def appendSelfPlayDetail(self, time:float, score:int, generation:int, game_id:int, id:int):
    self.self_play_detail.loc[(generation, id),:] = [time, score, game_id]

  def appendSelfPlayBroad(self, time:float, mean_score:float):
    current_index = len(self.self_play_broad.index.values)
    self.self_play_broad.loc[current_index] = [time, mean_score]

  def appendTraining(self, time:float, num_minibatch:int, mean_time_per_minibatch:float):
    current_index = len(self.training.index.values)
    self.training.loc[current_index] = [time, num_minibatch, mean_time_per_minibatch]

  def loadAllDataFrames(self, generation):
    self.self_play_detail = pd.read_pickle(r'C:\Users\Chris Minar\Documents\Python\Snake\saves\sp_detail_gen{}.pkl'.format(generation))
    self.self_play_broad  = pd.read_pickle(r'C:\Users\Chris Minar\Documents\Python\Snake\saves\sp_broad_gen{}.pkl'.format(generation))
    self.training         = pd.read_pickle(r'C:\Users\Chris Minar\Documents\Python\Snake\saves\train_gen{}.pkl'.format(generation))

  def saveAllDataFrames(self, generation):
    self.self_play_detail.to_pickle(r'C:\Users\Chris Minar\Documents\Python\Snake\saves\sp_detail_gen{}.pkl'.format(generation))
    self.self_play_broad.to_pickle( r'C:\Users\Chris Minar\Documents\Python\Snake\saves\sp_broad_gen{}.pkl'.format(generation))
    self.training.to_pickle(        r'C:\Users\Chris Minar\Documents\Python\Snake\saves\train_gen{}.pkl'.format(generation))