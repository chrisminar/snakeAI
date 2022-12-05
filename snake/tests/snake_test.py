"""Test functionality of snake base class."""

import numpy as np
import pytest

from snake.snake import Direction, GridEnum, Snake
from training.helper import GRID_X, GRID_Y


# pylint: disable=protected-access
class SnakeDummy(Snake):
    """Dummy child of Snake to test the functions of the base class."""

    def direction_to_tuple(self, direction):
        """Dummy."""
        if direction == Direction.UP:
            return 0, -1
        if direction == Direction.RIGHT:
            return 1, 0
        if direction == Direction.DOWN:
            return 0, -1
        if direction == Direction.LEFT:
            return -1, 0


@pytest.mark.parametrize("success, headx, heady, foodx, foody",
                         [(True, None, None, None, None),  # default reset
                          # reset
                          (True, 0,
                           0, 1, 1),
                          # don't pass food
                          (True, 0, 0,
                           None, None),
                          # head and food at same spot
                          (False, 0,
                           0, 0, 0),
                          # invalid head position
                          (False, -1,
                           0, 0, 0),
                          (False,
                           0, -1, 0, 0),
                          (False, GRID_X,
                           0, 0, 0),
                          (False, 0,
                           GRID_Y, 0, 0),
                          # invalid food position
                          (False, 0,
                           0, -1, 0),
                          (False, 0,
                           0, 0, -1),
                          (False, 0, 0,
                           GRID_X, 0),
                          (False, 0, 0, 0, GRID_Y)])
def test_reset(success: bool, headx: int, heady: int, foodx: int, foody: int) -> None:
    """Test grid reset function."""
    snake = SnakeDummy(GRID_X, GRID_Y)
    # reset grid
    snake.grid.fill(GridEnum.EMPTY.value)

    # place head
    snake.head_x = GRID_X-1
    snake.head_y = GRID_Y-1
    snake.grid[snake.head_y, snake.head_x] = GridEnum.HEAD.value

    # place food
    snake.food_x = GRID_X-2
    snake.food_y = GRID_Y-1
    snake.grid[snake.food_y, snake.food_x] = GridEnum.FOOD.value

    if success:
        if headx is None:
            snake._reset(food_x=foodx, food_y=foody)
        else:
            snake._reset(head_x=headx, head_y=heady,
                         food_x=foodx, food_y=foody)
        if headx is not None:
            assert headx == snake.head_x
        if heady is not None:
            assert heady == snake.head_y
        assert snake.grid[snake.head_y, snake.head_x] == GridEnum.HEAD.value
        if foodx is not None:
            assert foodx == snake.food_x
        if foody is not None:
            assert foody == snake.food_y
        assert snake.grid[snake.food_y, snake.food_x] == GridEnum.FOOD.value
    else:
        with pytest.raises(ValueError):
            snake._reset(head_x=headx, head_y=heady,
                         food_x=foodx, food_y=foody)


def test_init() -> None:
    """Test snake initialization."""
    snake = SnakeDummy(GRID_X, GRID_Y)
    assert not snake.game_over
    assert snake.grid.shape == np.zeros((GRID_X, GRID_Y)).shape


@pytest.mark.parametrize("food", [True, False])
@pytest.mark.parametrize("head_x, head_y, x_dir, y_dir", [(0, 0, 1, 0),
                                                          (1, 0, -1, 0),
                                                          (0, 0, 0, 1),
                                                          (0, 1, 0, -1)])
def test_run_single_no_food(food: bool, head_x: int, head_y: int, x_dir: int, y_dir: int) -> None:
    """Run single step, do not eat."""
    snake = SnakeDummy(GRID_X, GRID_Y)

    food_x = GRID_X-1 if not food else head_x+x_dir
    food_y = GRID_Y-1 if not food else head_y+y_dir
    snake._reset(head_x=head_x, head_y=head_y, food_x=food_x, food_y=food_y)

    x_old = snake.head_x
    y_old = snake.head_y
    snake.run_single(x_dir, y_dir)  # move right
    assert snake.head_x == x_old+x_dir
    assert snake.head_y == y_old+y_dir
    if food:
        assert snake.length == 1
    else:
        assert snake.length == 0


def test_spawn_food() -> None:
    """Test that food can be spawned at a random valid location."""
    snake = SnakeDummy(GRID_X, GRID_Y)
    num = 100
    food_x = np.zeros((num,))
    food_y = np.zeros((num,))

    for food in range(100):
        snake.grid.fill(GridEnum.EMPTY.value)
        snake.grid[0, :] = GridEnum.HEAD.value  # fill first two rows with 0
        snake.grid[1, :] = GridEnum.HEAD.value
        food_x[food], food_y[food] = snake.spawn_food()
    assert np.all(food_x > 1)  # no food values where body is
    assert np.all(food_y >= 0)  # within legal bounds
    assert np.all(food_y < snake.grid_size_y)
    assert np.all(food_x < snake.grid_size_x)
    # not pumping out the same value every time
    assert len(np.unique(food_x)) > 1
    # not pumping out the same value every time
    assert len(np.unique(food_y)) > 1


@pytest.mark.parametrize("direction", list(Direction))
def test_check_game_over(direction: Direction) -> None:
    """Move snake right until game over."""
    snake = SnakeDummy(GRID_X, GRID_Y)
    x_dir, y_dir = snake.direction_to_tuple(direction)

    head_x = GRID_X-1 if x_dir == 1 else 0
    head_y = GRID_Y-1 if y_dir == 1 else 0
    snake._reset(head_x=head_x, head_y=head_y)
    snake.run_single(x_dir, y_dir)
    assert snake.game_over


def test_check_game_over_tail() -> None:
    """Move snake into tail to end game."""
    # init
    snake = SnakeDummy(4, 4)
    #-2 -1
    # 0 -1

    snake._reset(head_x=0, head_y=0, food_x=0, food_y=1)

    # move up and eat
    # 0 -2
    # 1 -1
    snake.run_single(0, 1)  # move up
    snake.food_x = snake.food_y = 1   # place food at (1,1)
    snake.grid[snake.food_y, snake.food_x] = GridEnum.FOOD.value  # set food

    # move right and eat
    # 1  0
    # 2 -2
    snake.run_single(1, 0)
    snake.food_x, snake.food_y = 1, 0  # place food at (1,0)
    snake.grid[snake.food_y, snake.food_x] = GridEnum.FOOD.value

    # move down and eat
    # 2  1
    # 3  0
    snake.run_single(0, -1)

    # move left into old tail spot
    # 3  2
    # 0  1
    snake.run_single(-1, 0)
    assert not snake.game_over

    # move right into tail
    # -1 3
    #  1 2/0
    snake.run_single(1, 0)
    assert snake.game_over
