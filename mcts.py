from neuralNet import neural_network

class mcts:
  def __init__(self, state:game_state, neural_net:neural_network):
    self.s = state
    self.f_theta = neural_net
    self.root = mcts_node(self.s)

  def evaluate(self):
    f_theta.evaluate(self.s)
    return self.f_theta.out.move

class mcts_node:
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
