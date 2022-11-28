import unittest

import numpy as np

from helper import GRID_X, GRID_Y
from neural_net import NeuralNetwork
from play_games import PlayGames
from snake.snake import Snake
from snake.snake_reinforcement_learning import SnakeRL


class Snake_test(unittest.TestCase):
    def test_init(self):
        s = Snake(GRID_X, GRID_Y)
        self.assertEqual(s.gameover, False)
        self.assertEqual(s.grid.shape, np.zeros(
            (GRID_X, GRID_Y)).shape)

    def test_run_single_no_food(self):
        s = Snake(GRID_X, GRID_Y)
        x = s.head_x
        y = s.head_y
        l = s.length
        s.runSingle(1, 0)  # move right
        self.assertEqual(s.X, x+1)
        self.assertEqual(s.Y, y)
        self.assertEqual(s.length, 0)
        x = s.head_x
        y = s.head_y
        l = s.length
        s.runSingle(-1, 0)  # move left
        self.assertEqual(s.X, x-1)
        self.assertEqual(s.Y, y)
        self.assertEqual(s.length, 0)
        x = s.head_x
        y = s.head_y
        l = s.length
        s.runSingle(0, 1)  # move up
        self.assertEqual(s.X, x)
        self.assertEqual(s.Y, y+1)
        self.assertEqual(s.length, 0)
        x = s.head_x
        y = s.head_y
        l = s.length
        s.runSingle(0, -1)  # move down
        self.assertEqual(s.X, x)
        self.assertEqual(s.Y, y-1)
        self.assertEqual(s.length, 0)

    def test_run_single_with_food(self):
        s = Snake(GRID_X, GRID_Y)
        x = s.X
        y = s.Y
        l = s.length
        score = s.score
        s.foodX = x+1
        s.foodY = y
        s.grid[s.foodX][s.foodY] = -2
        s.runSingle(1, 0)  # move right
        self.assertEqual(s.grid[x][y], l+1)
        self.assertGreater(s.score, score)
        self.assertEqual(s.length, l+1)

    def test_spawn_food(self):
        s = Snake(GRID_X, GRID_Y)
        i = np.zeros((10,))
        j = np.zeros((10,))
        for food in range(10):
            i[food], j[food] = s.spawn_food()
        self.assertEqual(np.amax(s.grid), 0)  # check no food
        self.assertGreater(np.std(i), 0)
        self.assertGreater(np.std(j), 0)

    def test_check_game_over_right(self):
        s = Snake(GRID_X, GRID_Y)
        # head at 0,0
        s.runSingle(1, 0)  # head at 1,0
        self.assertFalse(s.gameover)
        s.runSingle(1, 0)  # head at 2,0
        self.assertFalse(s.gameover)
        s.runSingle(1, 0)  # head at 3,0
        self.assertFalse(s.gameover)
        s.runSingle(1, 0)  # head at 4,0 (dead)
        self.assertTrue(s.gameover)

    def test_checkGameOver_left(self):
        s = Snake(GRID_X, GRID_Y)
        # head at 0,0
        s.runSingle(-1, 0)  # head at -1,0
        self.assertTrue(s.gameover)

    def test_checkGameOver_up(self):
        s = Snake(GRID_X, GRID_Y)
        # head at 0,0
        s.runSingle(0, 1)  # head at 0,1
        self.assertFalse(s.gameover)
        s.runSingle(0, 1)  # head at 0,2
        self.assertFalse(s.gameover)
        s.runSingle(0, 1)  # head at 0,3
        self.assertFalse(s.gameover)
        s.runSingle(0, 1)  # head at 0,4 (dead)
        self.assertTrue(s.gameover)

    def test_checkGameOver_down(self):
        s = Snake(GRID_X, GRID_Y)
        # head at 0,0
        self.assertFalse(s.gameover)
        s.runSingle(0, -1)  # head at 0,-1
        self.assertTrue(s.gameover)

    def test_checkGameOver_tail(self):
        s = Snake(GRID_X, GRID_Y)
        s.grid[0][1] = -2  # place food at (0,1)
        s.foodX, s.foodY = 0, 1
        s.runSingle(0, 1)  # move up and eat
        s.grid[1][1] = -2  # place food at (1,1)
        s.foodX, s.foodY = 1, 1
        s.runSingle(1, 0)  # move right and eat
        s.grid[1][0] = -2  # place food at (1,0)
        s.foodX, s.foodY = 1, 0
        s.runSingle(0, -1)  # move down and eat
        s.runSingle(-1, 0)  # move left into old tail spot
        self.assertFalse(s.gameover)
        s.runSingle(1, 0)  # move right into tail
        self.assertTrue(s.gameover)


class SnakeRL_test(unittest.TestCase):
    def test_init(self):
        nn = NeuralNetwork()
        s = SnakeRL(nn=nn, sizeX=8, sizeY=8)
        self.assertTrue(hasattr(s, 'nn'))

    def test_Runstep_no_food(self):
        nn = NeuralNetwork()
        s = SnakeRL(nn=nn, sizeX=GRID_X, sizeY=GRID_Y)
        x = s.X
        y = s.Y
        l = s.length
        s.runStep(1)  # move right
        self.assertEqual(s.X, x+1)
        self.assertEqual(s.Y, y)
        self.assertEqual(s.length, l)
        x = s.X
        y = s.Y
        l = s.length
        s.runStep(3)  # move left
        self.assertEqual(s.X, x-1)
        self.assertEqual(s.Y, y)
        self.assertEqual(s.length, l)
        x = s.X
        y = s.Y
        l = s.length
        s.runStep(2)  # move up (up on grid, down if looking at it)
        self.assertEqual(s.X, x)
        self.assertEqual(s.Y, y+1)
        self.assertEqual(s.length, l)
        x = s.X
        y = s.Y
        l = s.length
        s.runStep(0)  # move down (down on grid, up i flooking at it)
        self.assertEqual(s.X, x)
        self.assertEqual(s.Y, y-1)
        self.assertEqual(s.length, l)

    def test_Runstep_with_food(self):
        nn = NeuralNetwork()
        s = SnakeRL(nn=nn, sizeX=GRID_X, sizeY=GRID_Y)
        x = s.X
        y = s.Y
        l = s.length
        score = s.score
        s.foodX = x+1
        s.foodY = y
        s.grid[s.foodX][s.foodY] = -2
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
        self.assertTrue(s.gameover, False)
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


if __name__ == '__main__':
    unittest.main()
