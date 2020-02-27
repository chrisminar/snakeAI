from neuralNet import NeuralNetwork
from gameState import GameState


class Mcts:
  def __init__(self, state:GameState, neural_net:NeuralNetwork):
    self.s = state
    self.f_theta = neural_net
    self.root = mcts_node(self.s)

  def evaluate(self):
    f_theta.evaluate(self.s)
    return self.f_theta.out.move

class Mcts_node:
  #mcts - node
  # game state
  # P (s,a) float[]
  # vist count N (s,a) int
  # action value Q(s,a) float
  # children mctsNode[]
  # parent mctsNode
  def __init__(self, state, parent = None):
    self.parent = parent
    self.s = state
    self.P = [0.0]*4
    self.N = 0
    self.Q = 0.0
    self.children = []
