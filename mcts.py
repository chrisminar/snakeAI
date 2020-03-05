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
    P = (1-globe.MCTS_EPSILON)*root.P + globe.MCTS_EPSILON * np.random.dirichlet(np.zeros((4,)) + globe.MCTS_DIR) #Noise
    self.root = Mcts_node(self.s, policy)
    Mcts.addMoveSpace(self.root.s, self.root)

    while 1:
      leaf = self.select(self.root)
      newLeaf, v = self.expand(leaf)
      self.backup(newLeaf, v)
      self.moves += 1
      if (self.moves > globe.NUM_MCTS_ITER) or Mcts.itsDead(self.root): #end condition
        break;

    return selectFinal()

  def select(self, root:Mcts_node): #PAGE 26
    """Select a leaf -- sometimes with the highest action value"""
    maxedVal = 0
    index = 0
    if not root.isLeaf:
      sumVisitCount = np.sqrt(sum(i.N for i in root.children))
      for i, child in enumerate(root.children):
        U = globe.CPUCT * root.P[i] * sumVisitCount / (1+child.N)
        a_t = U + child.Q
        if a_t > maxedVal:
          maxedVal = a_t
          index = i
      self.select(root.children(index))
    else:
      return root

  def expand(self, leaf:Mcts_node):
    """Add a new leaf to the tree given a leaf"""
    # add move space to leaf
    Mcts.addMoveSpace(leaf.s, leaf)
    # run neural network
    policy, value = self.f_theta.evaluate(leaf.s)
    # add move
    direction = np.argmax(policy)
    leaf.P = policy
    leaf.isLeaf = False
    return leaf, value

  def backup(self, leaf:Mcts_node, v):
    """Work up the tree from a given and update values"""
    leaf.N += 1
    leaf.W += v
    leaf.Q = leaf.W/leaf.N
    if leaf.parent == None: #no parent, root reached
      return
    else:
      self.backup(leaf.parent, v)

  def selectFinal(self):
    max = 0
    idx = 0
    sumVisitCount = np.sqrt(sum(i.N**(1/self.temp) for i in self.root.children))
    for i, child in enumerate(self.root.children):
      vcount = child.N**(1/self.temp) / sumVistcount
      if vcount > max:
        max = vcount
        idx = i

    self.root = Mcts.prune(self.root, i)
    count = Mcts.counter(self.root, 0)
    return i, count

  def prune(root:Mcts_node, move:int):
    for child in reversed(root.children):
      if (child.direction == move):
        pass
      else:
        del root.children[i]
    return root.children[0]

  def counter(root:Mcts_node, s:int):
    for child in root.children:
      if not child.isLeaf:
        s += 1
        s = Mcts.counter(s)
      else:
        return s

  def itsDead(root):
    for child in root.children():
      if child.isDead == False:
        return False
    return True

  def legalMoves(state):
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
    return isLegal

  def addMoveSpace(state:GameState, root:Mcts_node):
    for direction in range(4):
      grid, score, gameover = GameState.move(root.s, direction, root.score)
      root.append(Mcts_node(grid, direction = direction, parent = root, score = score, isDead = gameover))

class Mcts_node:
  #mcts - node
  # game state
  # P (s,a) float[]
  # vist count N (s,a) int
  # action value Q(s,a) float
  # children mctsNode[]
  # parent mctsNode
  def __init__(self, state, probability = np.zeros((4,)), direction:int = 0, parent:Mcts_node = None, score:int = 0, isDead:bool=False, isLeaf:bool=True):

    self.parent = parent       # parent mcts node
    self.children = []         # children mcts nodes

    self.s = state             # snake game state
    self.direction = direction # direction moved to get to this state
    self.score = score         # snake score

    self.isDead = isDead       # has the game ended at this state?
    self.isLeaf = isLeaf       # is this a leaf node?

    self.P = probability       # move probabilities at this state
    self.N = 0                 # visit count
    self.W = 0                 # total action value
    self.Q = 0.0               # mean action value
    
    


