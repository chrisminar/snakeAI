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
    """Select a leaf with the highest action value"""
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

  def expand(self, leaf:Mcts_node):
    """Add a new leaf to the tree given a leaf"""
    # make move
    grid, score, direction, gameover = GameState.move(leaf.s, leaf.P, leaf.score)
    # run neural network
    policy, value = self.f_theta.evaluate(grid)
    # add move
    leaf.children.append(Mcts_node(grid, policy, parent=leaf, direction=direction, score=score, isDead=gameover))
    return leaf.children[-1]

  def backup(self, leaf:Mcts_node, v):
    """Work up the tree from a given and update values"""
    leaf.N += 1
    leaf.W = v
    leaf.Q = leaf.W/leaf.N
    if leaf.parent == None: #no parent, root reached
      return
    else:
      self.backup(leaf.parent)

  def prune(self, root:Mcts_node, move:int):
    for child in reversed(root.children):
      if (child.direction == move):
        pass
      else:
        del root.children[i]
    return

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
  def __init__(self, state, probability = np.zeros((4,)), direction = dicrection, parent = None, score = 0, isDead=False):
    self.parent = parent # parent mcts node
    self.s = state # snake game state
    self.score = score # snake score
    self.direction = direction # direction moved to get to this state
    self.isDead = isDead # has the game ended at this state?
    self.P = probability # move probabilities at this state
    self.N = 0 #visit count
    self.W = 0 #total action value
    self.Q = 0.0 #mean action value
    self.children = [] #children mcts nodes
    


