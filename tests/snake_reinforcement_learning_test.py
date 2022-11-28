"""Test functionality of rl snake class."""

import numpy as np

from helper import GRID_X, GRID_Y
from neural_net import NeuralNetwork
from play_games import PlayGames
from snake.snake_reinforcement_learning import SnakeRL


def test_init(self):
    nn = NeuralNetwork()
    s = SnakeRL(nn=nn, sizeX=8, sizeY=8)
    self.assertTrue(hasattr(s, 'nn'))


def test_Runstep_no_food(self):
    nn = NeuralNetwork()
    s = SnakeRL(nn=nn, sizeX=GRID_X, sizeY=GRID_Y)
    x = s.head_x
    y = s.head_y
    l = s.length
    s.runStep(1)  # move right
    self.assertEqual(s.head_x, x+1)
    self.assertEqual(s.head_y, y)
    self.assertEqual(s.length, l)
    x = s.head_x
    y = s.head_y
    l = s.length
    s.runStep(3)  # move left
    self.assertEqual(s.head_x, x-1)
    self.assertEqual(s.head_y, y)
    self.assertEqual(s.length, l)
    x = s.head_x
    y = s.head_y
    l = s.length
    s.runStep(2)  # move up (up on grid, down if looking at it)
    self.assertEqual(s.head_x, x)
    self.assertEqual(s.head_y, y+1)
    self.assertEqual(s.length, l)
    x = s.head_x
    y = s.head_y
    l = s.length
    s.runStep(0)  # move down (down on grid, up i flooking at it)
    self.assertEqual(s.head_x, x)
    self.assertEqual(s.head_y, y-1)
    self.assertEqual(s.length, l)


def test_Runstep_with_food(self):
    nn = NeuralNetwork()
    s = SnakeRL(nn=nn, sizeX=GRID_X, sizeY=GRID_Y)
    x = s.head_x
    y = s.head_y
    l = s.length
    score = s.score
    s.food_x = x+1
    s.food_y = y
    s.grid[s.food_x][s.food_y] = -2
    s.runStep(1)  # move right
    self.assertEqual(s.grid[x][y], l+1)
    self.assertGreater(s.score, score)
    self.assertEqual(s.length, l+1)


def test_evaluateNext(self):
    nn = NeuralNetwork()
    s = SnakeRL(nn=nn, sizeX=GRID_X, sizeY=GRID_Y)
    g = PlayGames(nn)
    direction, move_array, head = s.evaluateNextStep(g.gamestate_to_nn)
    print(direction)
    self.assertGreaterEqual(np.argmax(direction), 0,
                            'invalid direction output')
    self.assertLessEqual(np.argmax(move_array), 3,
                         'invalid direction output')
    self.assertEqual(np.argmax(move_array), direction,
                     'direction doesn\'t match move array')
    self.assertEqual(head[0], 1, 'up is free')
    self.assertEqual(head[1], 1, 'right is free')
    self.assertEqual(head[2], 0, 'down is free')
    self.assertEqual(head[3], 0, 'left is free')


def test_play(self):
    nn = NeuralNetwork()
    s = SnakeRL(nn=nn, sizeX=GRID_X, sizeY=GRID_Y)
    g = PlayGames(nn)
    s.play(g.gamestate_to_nn)
    self.assertTrue(s.game_over, False)
    self.assertGreater(len(s.moveList), 0)
    print(s.moveList)
    print(s.grid)

# test converthead at starting position


def test_convertHead_start(self):
    # no food
    grid = np.zeros((4, 4))-1  # empty
    grid[0, 0] = 0  # head
    isFree = SnakeRL.convertHead(0, 0, 4, 4, grid)
    truth = [1, 1, 0, 0]  # up and right are free
    for i, j in zip(isFree, truth):
        self.assertEqual(i, j)

    # with food
    grid[0, 1] = -2  # food
    isFree = SnakeRL.convertHead(0, 0, 4, 4, grid)
    for i, j in zip(isFree, truth):
        self.assertEqual(i, j)

# test converthead while along top wall


def test_convertHead_top(self):
    # no food
    grid = np.zeros((4, 4))-1  # empty
    grid[1, 3] = 0  # head
    isFree = SnakeRL.convertHead(1, 3, 4, 4, grid)
    truth = [0, 1, 1, 1]  # up not free
    for i, j in zip(isFree, truth):
        self.assertEqual(i, j)

    # with food
    grid[1, 2] = -2  # food
    isFree = SnakeRL.convertHead(1, 3, 4, 4, grid)
    for i, j in zip(isFree, truth):
        self.assertEqual(i, j)

# test converthead while along right wall


def test_convertHead_right(self):
    # no food
    grid = np.zeros((4, 4))-1  # empty
    grid[3, 1] = 0  # head
    isFree = SnakeRL.convertHead(3, 1, 4, 4, grid)
    truth = [1, 0, 1, 1]  # up not free
    for i, j in zip(isFree, truth):
        self.assertEqual(i, j)

    # with food
    grid[2, 1] = -2  # food
    isFree = SnakeRL.convertHead(3, 1, 4, 4, grid)
    for i, j in zip(isFree, truth):
        self.assertEqual(i, j)
