"""Parallelizable snake."""

from typing import Final, List, Optional, Tuple

import numpy as np
from numpy import typing as npt

from training.helper import (EXPLORATORY_MOVE_FRACTION, GRID_X, GRID_Y,
                             MAXIMUM_MOVES_WITHOUT_EATING, MAXIMUM_TOTAL_MOVES,
                             NUM_SELF_PLAY_GAMES, SCORE_FOR_GAME_WIN,
                             SCORE_PENALTY_FOR_FAILURE, SCORE_PER_FOOD,
                             SCORE_PER_MOVE, Direction, GridEnum, Timer,
                             grid_2_nn)
from training.neural_net import NeuralNetwork

_RNG: Final = np.random.default_rng()


class ParSnake:
    """Parallel snake class."""

    def __init__(self, neural_net: NeuralNetwork, grid_size_x: int = GRID_X, grid_size_y: int = GRID_Y, exploratory: bool = False, num_games: int = NUM_SELF_PLAY_GAMES) -> None:
        """_summary_

        Args:
            grid_size_x (int, optional): _description_. Defaults to GRID_X.
            grid_size_y (int, optional): _description_. Defaults to GRID_Y.
            exploratory (bool, optional): _description_. Defaults to False.
        """
        self.exploratory = exploratory

        # initialize grids
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.grid = np.full((num_games, self.grid_size_y,
                            self.grid_size_x), GridEnum.EMPTY.value, dtype=np.int32)
        self.grid_size = self.grid_size_x*self.grid_size_y

        # initialize snakes
        valids = choose_valids(
            self.grid == GridEnum.EMPTY.value)
        self.heads_x, self.heads_y = valids[:, 1], valids[:, 0]
        self.grid0 = np.arange(num_games, dtype=np.int32)  # for indexing grid0
        self.lengths = np.zeros((num_games,), dtype=np.int32)
        self.grid[self.grid0, self.heads_y, self.heads_x] = GridEnum.HEAD.value
        # Last move taken (-1 means no move taken)
        self.last_move = np.full_like(self.lengths, -1)

        # scoring
        self.scores = np.zeros_like(self.lengths)
        self.num_moves = np.zeros_like(self.lengths)
        self.moves_since_food = np.zeros_like(self.lengths)

        # input
        self.direction_xs = np.zeros_like(self.lengths)
        self.direction_ys = np.zeros_like(self.lengths)

        # gamestate
        self.games_over = False

        # initialize food
        valids = choose_valids(
            self.grid == GridEnum.EMPTY.value)
        self.food_xs, self.food_ys = valids[:, 1], valids[:, 0]
        self.grid[self.grid0, self.food_ys, self.food_xs] = GridEnum.FOOD.value

        # init neural network
        self.neural_net = neural_net

        # init tracking
        self.states: List[npt.NDArray[np.int32]] = []
        self.heads: List[npt.NDArray[np.bool8]] = []
        self.game_id_tracker: List[npt.NDArray[np.int32]] = []
        self.moves: List[npt.NDArray[np.int32]] = []
        self.final_score = np.zeros_like(self.lengths)
        self.game_ids = np.arange(num_games, dtype=np.int32)

    def aggregate_results(self) -> Tuple[npt.NDArray[np.int32],
                                         npt.NDArray[np.bool8],
                                         npt.NDArray[np.int32],
                                         npt.NDArray[np.int32],
                                         npt.NDArray[np.int32]]:
        """Aggregate results of the games.

        Returns:
            (npt.NDArray[np.uint32]): state, shape (n, y, x)
            (npt.NDArray[npt.bool8]): head, shape (n, 4)
            (npt.NDArray[np.int32]): score, shape (n, )
            (npt.NDArray[np.int32]): game id, shape (n, )
            (npt.NDArray[np.int32]): move, shape (n, 4)
        """
        game_state = np.concatenate(self.states, axis=0)
        head_view = np.concatenate(self.heads, axis=0)
        game_id = np.concatenate(self.game_id_tracker, axis=0)
        game_score = self.final_score[game_id]
        move_made = np.concatenate(self.moves, axis=0)
        assert game_state.shape[0] == head_view.shape[0] == game_score.shape[0] == game_id.shape[0] == move_made.shape[0]
        return game_state, head_view, game_score, game_id, move_made

    def step_time(self, directions: npt.NDArray[np.int32]):
        """Move the snakes one step forward.

        Args:
            directions (npt.NDArray[np.int32]): Direction for each snake to move.
        """
        self.last_move = directions.copy()

        self._snake_head_tracker_update(directions)

        if self.games_over:
            return

        self.scores += SCORE_PER_MOVE
        self.num_moves += 1

        self._snake_ate_this_turn()

        if self.games_over:
            return

        self._update_grid()

        self._spawn_food()

        self._update_head()

    def _snake_head_tracker_update(self, directions: npt.NDArray[np.int32]) -> None:
        """Move snake head trackers.

        Args:
            directions (npt.NDArray[np.int32]): Direction to move snake.
        """
        xy_array = directions_to_tuples(directions)
        self.direction_xs = xy_array[:, 0]
        self.direction_ys = xy_array[:, 1]
        self.heads_x += self.direction_xs
        self.heads_y += self.direction_ys
        games_over = self.check_game_over()
        self.eject_games(games_to_eject=games_over)

    def _snake_ate_this_turn(self) -> None:
        """Deal with snake eating this step."""
        self.moves_since_food += 1
        ate_this_turn = np.logical_and(
            self.heads_x == self.food_xs,  self.heads_y == self.food_ys)
        self.lengths[ate_this_turn] += 1
        self.scores[ate_this_turn] += SCORE_PER_FOOD
        self.moves_since_food[ate_this_turn] = 0

        games_won = self.lengths == self.grid_size-1
        self.scores[games_won] += SCORE_FOR_GAME_WIN
        self.eject_games(games_to_eject=games_won)

    def _update_grid(self) -> None:
        """Update grid for move.
        """
        did_not_eat = np.logical_not(np.logical_and(
            self.heads_x == self.food_xs,  self.heads_y == self.food_ys))
        self.grid[self.grid >= GridEnum.HEAD.value] += 1
        for hungry, grid, length in zip(did_not_eat, self.grid, self.lengths):
            if hungry:
                grid[grid > length] = GridEnum.EMPTY.value

    def _spawn_food(self) -> None:
        ate_this_turn = (self.heads_x == self.food_xs) & (
            self.heads_y == self.food_ys)
        yx_index = choose_valids(
            self.grid[ate_this_turn] == GridEnum.EMPTY.value)
        self.food_xs[ate_this_turn] = yx_index[:, 1]
        self.food_ys[ate_this_turn] = yx_index[:, 0]
        self.grid[ate_this_turn, yx_index[:, 0],
                  yx_index[:, 1]] = GridEnum.FOOD.value

    def _update_head(self) -> None:
        games_over = self.check_game_over()
        self.scores[games_over] += SCORE_PENALTY_FOR_FAILURE
        self.eject_games(games_to_eject=games_over)
        self.grid[self.grid0, self.heads_y, self.heads_x] = GridEnum.HEAD.value

    def eject_games(self, *, games_to_keep: Optional[npt.NDArray[np.bool8]] = None,
                    games_to_eject: Optional[npt.NDArray[np.bool8]] = None) -> None:
        """Eject games to be processed.

        Args:
            games_to_keep (Optional[npt.NDArray[np.bool8]], optional): Games to keep. Defaults to None.
            games_to_eject (Optional[npt.NDArray[np.bool8]], optional): Games to not keep. Defaults to None.

        Raises:
            ValueError: Can only pass one of the two input args.
        """
        if (games_to_keep is None and games_to_eject is None) or (games_to_keep is not None and games_to_eject is not None):
            raise ValueError("Must pass games to keep or games to eject")
        if games_to_eject is not None:
            games_to_keep = np.logical_not(games_to_eject)
        assert games_to_keep is not None
        assert games_to_keep.ndim == 1

        # update final score
        self.final_score[self.game_ids] = self.scores

        # boot old values
        self.game_ids = self.game_ids[games_to_keep]
        self.grid0 = np.arange(np.unique(self.game_ids).size)
        self.grid = self.grid[games_to_keep]
        self.heads_x = self.heads_x[games_to_keep]
        self.heads_y = self.heads_y[games_to_keep]
        self.lengths = self.lengths[games_to_keep]
        self.last_move = self.last_move[games_to_keep]
        self.scores = self.scores[games_to_keep]
        self.num_moves = self.num_moves[games_to_keep]
        self.moves_since_food = self.moves_since_food[games_to_keep]
        self.direction_xs = self.direction_xs[games_to_keep]
        self.direction_ys = self.direction_ys[games_to_keep]
        self.food_xs = self.food_xs[games_to_keep]
        self.food_ys = self.food_ys[games_to_keep]
        self.check_all_games_over()

    def play(self) -> None:
        """Play all snake games till completion."""
        with Timer("Played games"):
            while not self.games_over:
                new_directions, new_moves, new_heads = self.evaluate()
                self.states.append(self.grid.copy())
                self.moves.append(new_moves.copy())
                self.heads.append(new_heads.copy())
                self.game_id_tracker.append(self.game_ids.copy())
                self.step_time(new_directions)

    def evaluate(self) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.bool8]]:
        """Evaluate the current set of snake games.

        Returns:
            Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.bool8]]:
                new_dir: (num_games, ) new directions for each game
                next_direction_array: (num_games, 4) new direction bools for each direction 1 = chosen
                heads: (num_games,4) which directions the head can safely move
        """
        pre_processed_grid = grid_2_nn(self.grid)
        heads = self.convert_heads()
        policy = self.neural_net.evaluate(state=pre_processed_grid, head=heads)

        # account for invalid choices from NN
        # true if no valid options from NN
        invalid_choice = np.all(policy == 0, axis=1)
        policy[invalid_choice, :] = heads[invalid_choice, :]

        # check that no invalid choices were handled from head
        invalid_choice = np.all(policy == 0, axis=1)
        # force up when no valid options
        # make this the largest option so it will get chosen in argmax
        policy[invalid_choice, Direction.UP.value] = np.inf

        # random moves
        new_dir = np.argmax(policy, axis=1).astype(int)
        if self.exploratory:
            random_idx = np.random.rand(
                *new_dir.shape) < EXPLORATORY_MOVE_FRACTION
            new_dir[random_idx] = choose_valids(
                heads[random_idx])[:, 1]

        # make an array of [up, right, left, down] for each move where the chosen direction is 1 and others are 0
        next_direction_array = np.zeros((new_dir.size, 4), dtype=np.int32)
        np.put_along_axis(next_direction_array, np.expand_dims(
            new_dir.copy(), axis=1), 1, axis=1)

        return new_dir, next_direction_array, heads

    def check_all_games_over(self) -> bool:
        """Check if all the games are done playing.

        Returns:
            bool: If all games are done.
        """
        self.games_over = self.lengths.size == 0
        return self.games_over

    def check_game_over(self) -> npt.NDArray[np.bool8]:
        """Which games are over.

        Returns:
            npt.NDArray[np.bool8]: Is game over.
        """
        games_over = np.full((self.grid.shape[0],), False, dtype=np.bool8)
        games_over[self.moves_since_food >=
                   MAXIMUM_MOVES_WITHOUT_EATING] = True
        games_over[self.num_moves >= MAXIMUM_TOTAL_MOVES] = True

        games_over[self.heads_x < 0] = True
        x_bad = self.heads_x >= self.grid_size_x
        x_ok = np.logical_not(x_bad)
        games_over[x_bad] = True

        games_over[self.heads_y < 0] = True
        y_bad = self.heads_y >= self.grid_size_y
        y_ok = np.logical_not(y_bad)
        games_over[y_bad] = True

        okay = x_ok & y_ok  # have to do some sneaky stuff to not throw index errors
        games_over[okay][self.grid[self.grid0[okay], self.heads_y[okay], self.heads_x[okay]]
                         >= GridEnum.HEAD.value] = True
        return games_over

    def convert_heads(self) -> npt.NDArray[np.bool8]:
        """Convert grids into headviews.

        Returns:
            npt.NDArray[np.bool8]: (num_grid, 4) 0 if direction is not free, 1 otherwise
        """
        is_free = np.zeros((self.grid.shape[0], 4), dtype=np.bool8)

        # is left empty or food
        left_ok = np.logical_and(self.heads_x > 0,
                                 self.grid[self.grid0, self.heads_y,
                                           self.heads_x-1] < GridEnum.HEAD.value)
        left_ok = left_ok & (self.last_move != Direction.RIGHT.value)
        is_free[left_ok, Direction.LEFT.value] = 1

        # is right empty or food
        right_ok = self.heads_x < self.grid_size_x - 1
        right_ok[right_ok] = self.grid[self.grid0[right_ok], self.heads_y[right_ok],
                                       self.heads_x[right_ok]+1] < GridEnum.HEAD.value
        right_ok = right_ok & (self.last_move != Direction.LEFT.value)
        is_free[right_ok, Direction.RIGHT.value] = 1

        # is above empty or food
        up_ok = np.logical_and(self.heads_y > 0, self.grid[self.grid0, self.heads_y -
                                                           1, self.heads_x] < GridEnum.HEAD.value)
        up_ok = up_ok & (self.last_move != Direction.DOWN.value)
        is_free[up_ok, Direction.UP.value] = 1

        # is below empty or food
        down_ok = self.heads_y < self.grid_size_y - 1
        down_ok[down_ok] = self.grid[self.grid0[down_ok], self.heads_y[down_ok]+1,
                                     self.heads_x[down_ok]] < GridEnum.HEAD.value
        down_ok = down_ok & (self.last_move != Direction.UP.value)
        is_free[down_ok, Direction.DOWN.value] = 1

        return is_free

    def _reset(self,
               *,
               head_x: int = 0,
               head_y: int = 0,
               food_x: Optional[int] = None,
               food_y: Optional[int] = None) -> None:
        """Reset the snake to the passed parameters.

        This is for testing purposes.

        Args:
            head_x (int): Head x position.
            head_y (int): Head y position.
            food_x (int): Food x position.
            food_y (int): Food y position.
        """
        if head_x == food_x and head_y == food_y:
            raise ValueError("Head and food can't be on same spot.")
        # reset grid
        self.grid.fill(GridEnum.EMPTY.value)

        # place head
        self.heads_x.fill(head_x)
        self.heads_y.fill(head_y)
        if not 0 <= head_x < self.grid_size_x or not 0 <= head_y < self.grid_size_y:
            raise ValueError("Invalid head position.")
        self.grid[self.grid0, self.heads_y, self.heads_x] = GridEnum.HEAD.value

        # place food
        if food_x is None:
            food_x = self.grid_size_x-1
        if food_y is None:
            food_y = self.grid_size_y-1
        self.food_xs.fill(food_x)
        self.food_ys.fill(food_y)
        if not 0 <= food_x < self.grid_size_x or not 0 <= food_y < self.grid_size_y:
            raise ValueError("Invalid food position")
        self.grid[self.grid0, self.food_ys, self.food_xs] = GridEnum.FOOD.value


def choose_valids(array: npt.NDArray[np.bool8], backups: Optional[npt.NDArray[np.bool8]] = None) -> npt.NDArray[np.int32]:
    """Choose valid indices from a 3d array

    Args:
        array (npt.NDArray[np.bool8]): Array of size (m,n,k). Will choose a random valid element from (n,k) for each m

    Returns:
        npt.NDArray[np.int32]: random indicies (m,2)
    """
    if array.ndim < 2:
        raise ValueError("Requires at least 2 dimensional array.")

    indicies = np.full((array.shape[0], 2), 0, dtype=np.int32)

    if backups is not None:
        for index, (one_snake, backup) in enumerate(zip(array, backups)):
            indicies[index] = get_random_valid(one_snake, backup)
    else:
        for index, one_snake in enumerate(array):
            indicies[index] = get_random_valid(one_snake)

    return indicies


def get_random_valid(one_snake: npt.NDArray[np.bool8], backup: Optional[npt.NDArray[np.bool8]] = None) -> npt.NDArray[np.int32]:
    """Get a random valid value.

    Args:
        one_snake (npt.NDArray[np.bool8]): Array of size (n,k) Will choose a random valid element.

    Returns:
        npt.NDArray[np.int32]: Indices of random valid element
    """
    try:
        return _RNG.choice(np.argwhere(one_snake))
    except ValueError:
        if backup is not None:
            try:
                return _RNG.choice(np.argwhere(backup))
            except ValueError:
                pass
    return np.array([0, 0], dtype=np.int32)


def directions_to_tuples(directions: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    """Convert enum directions to x, y directions.

    Args:
        directions (npt.NDArray[np.int32]): direction enum

    Raises:
        ValueError: If num directions dimensions is bad.

    Returns:
        npt.NDArray[np.int32]: x y directions for each state
    """
    if directions.ndim != 1:
        raise ValueError("Input directions must be num_games,)")

    delta_directions = np.zeros((directions.shape[0], 2), dtype=np.int32)

    for direction in Direction:
        direction_index = direction == directions
        delta_directions[direction_index] = direction.to_x_y()

    return delta_directions
