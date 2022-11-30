"""Play snake game in GUI."""

from typing import Any, Tuple

import pygame as pg

from snake.snake import GridEnum, Snake


class SnakeHuman(Snake):
    """Human playable snake."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # init pygame
        pg.init()
        self.pygame_display = pg.display.set_mode(
            (self.grid_size_x*21, self.grid_size_y*21), 0, 32)  # init display
        pg.display.set_caption('Snake')
        self.pygame_display.fill((255, 255, 255))  # fill with white
        self.display_state()

    def direction_to_tuple(self, direction: Any) -> Tuple[int, int]:
        """Convert pygame event to delta x and delta y.

        Args:
            direction (Direction): Direction to move head.

        Returns:
            Tuple[int,int]: Direction to move head in x and y.
        """
        if direction == pg.K_LEFT:
            return -1, 0
        if direction == pg.K_RIGHT:
            return 1, 0
        elif direction == pg.K_DOWN:
            return 0, 1
        elif direction == pg.K_UP:
            return 0, -1
        raise ValueError("Invalid key")

    def play(self) -> int:
        """Play a game of snake.

        Returns:
            int: Score
        """
        while not self.game_over:
            events = pg.event.get()
            for event in events:
                if event.type == pg.KEYDOWN and event.key in [pg.K_LEFT, pg.k_RIGHT, pg.K_DOWN, pg.K_UP]:
                    self.direction_x, self.direction_y = self.direction_to_tuple(
                        event.key)
                    break
            self.run_single(self.direction_x, self.direction_y)
            self.display_state()
        return self.score

    def grid_num_2_color(self, num: int) -> Tuple[int, int, int]:
        """Change a grid value to a color tuple.

        Args:
            num (int): grid value

        Returns:
            Tuple[int, int, int]: rgb color
        """
        if num == GridEnum.FOOD.value:
            return (255, 0, 0)
        if num == GridEnum.HEAD.value:
            return (0, 0, 0)
        if num == GridEnum.EMPTY.value:
            return (200, 200, 200)
        return (100, 100, 100)

    def display_state(self) -> None:
        """Draw the most recent frame."""
        for i in range(self.grid_size_x):
            for j in range(self.grid_size_y):
                pg.draw.rect(self.pygame_display, self.grid_num_2_color(
                    self.grid[i, j]), (i*21, j*21, 20, 20))
        pg.display.update()


if __name__ == "__main__":
    s = SnakeHuman()
    print(s.play())
