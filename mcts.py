from neuralNet import NeuralNetwork
from gameState import GameState
import numpy as np
from globalVar import Globe as globe

class Mcts:
  def __init__(self, state:GameState, nn:NeuralNetwork, temp=0.01):
    self.s = state
    self.f_theta = nn
    self.moves = 0
    self.temp = temp

  def mcts_search(self):
    policy = self.f_theta.evaluate(self.s)
    self.root = Mcts_node(self.s, policy)

    while 1:
      leaf = self.select(self.root)
      newLeaf = self.expand(leaf)
      self.backup(newLeaf)
      self.moves += 1
      if (self.moves > globe.NUM_MCTS_ITER) or self.itsDead(self.root): #end condition
        break;

    return selectFinal()

  def itsDead(self, root):
    for child in root.children():
      if child.isDead == False:
        return False
    return True

  def select(self, root:Mcts_node): #PAGE 26
    maxedVal = 0
    index = 0
    if len(root.children) > 0:
      for i, child in enumerate(root.children()):
        P = (1-globe.MCTS_EPSILON)*pca + globe.MCTS_EPSILON*globe.MCTS_DIR #HERE
        U = c_puct*child.P + sqrt()/(1+child.N) #HERE
        upperConfidenceBound = U + child.Q
        if upperConfidenceBound > maxedVal:
          maxedVal = upperConfidenceBound
          index = i
      self.select(root.children(index))
    else:
      return root

  def expand(self, root:Mcts_node):
    # run neural network
    policy, value = self.f_theta.evaluate(root.s)
    # make move
    newState = root.makeMove(policy)
    # add move
    root.children.append(Mcts_node(newState, policy, parent=root))
    return root.children[-1]

  def backup(self, root:Mcts_node, v):
    root.N += 1
    root.W = v
    root.Q = root.W/root.N
    if root.parent == None:
      return
    else:
      self.backup(root.parent)

  def selectFinal(self):
    max = 0
    idx = 0
    for i, child in enumerate(self.root.children):
      numerator = child.N**(1/self.temp)
      denom = 0
      for j, child2 in enumerate(self.root.children):
        if i not j:
          denom += child2.N**(1/self.temp)
      sel = numerator/denom
      if sel > max:
        max = sel
        idx = i

    self.prune(i)
    return 

  def legalMoves(self,state):
    isLegal = np.zeros((4,), dtype=bool)
    head = np.argwhere(state == 0)[0]
    x = head[0]
    y = head[1]
    if y < globe.GRID_Y - 1: #up legal
      isLegal[0] = True
    if y > 1: #down legal
      isLegal[2] = True
    if x > globe.GRID_X -1: #right legal
      isLegal[1] = True
    if x > 1: #up legal
      isLegal[3] = True

class Mcts_node:
  #mcts - node
  # game state
  # P (s,a) float[]
  # vist count N (s,a) int
  # action value Q(s,a) float
  # children mctsNode[]
  # parent mctsNode
  def __init__(self, state, probability = np.zeros((4,)), parent = None):
    self.parent = parent
    self.s = state
    self.P = probability #prior probability
    self.N = 0 #visit count
    self.W = 0 #total action value
    self.Q = 0.0 #mean action value
    self.children = []
    


