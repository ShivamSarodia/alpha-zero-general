import numpy as np

class Generals:
    def __init__(self, width, height):
        # step number (step / 2 = turn)
        # troops: how many troops on each square
        # who owns: + or -
        # features: mountains, city, general, etc.

        self.width = width
        self.height = height

        self.step = 0
        self.player = 1
        self.game_over = False

        # number of troops on each tile
        self.troops = np.zeros((width, height))

        # Who owns each tile. +1 and -1 are players, 0 is unowned.
        self.owns = np.zeros((width, height))

        # 0 is normal, 1 is general.
        # TODO: add cities and mountains
        self.features = np.zeros((width, height))

        # TODO: place features randomly
        self.features[0,0] = 1
        self.owns[0,0] = 1

        self.features[width-1, height-1] = 1
        self.owns[width-1, height-1] = -1

        self._grow_troops()

    def _update_game(self):
        capitals = self.owns[self.features==1]
        if capitals[0] == self.player and capitals[1] == self.player:
            self.game_over = True
            return

        self.player *= -1
        self.step += 1
        self._grow_troops()

    def _grow_troops(self):
        if self.step % 2 == 0:
            self.troops += (self.features==1)

    def _make_move(self, square, direction):
        if self.owns[square] != self.player:
            return

        if direction == "u":
            new_square = (square[0] - 1, square[1])
        elif direction == "l":
            new_square = (square[0], square[1] - 1)
        elif direction == "d":
            new_square = (square[0] + 1, square[1])
        elif direction == "r":
            new_square = (square[0], square[1] + 1)
        else:
            raise ValueError

        if not 0 <= new_square[0] < self.height:
            return
        if not 0 <= new_square[1] < self.width:
            return

        moved_troops = self.troops[square] - 1
        if moved_troops <= 0:
            return

        if self.owns[new_square] == -self.player:
            if self.troops[new_square] >= moved_troops:
                self.troops[new_square] -= moved_troops
            else:
                self.troops[new_square] = moved_troops - self.troops[new_square]
                self.owns[new_square] *= -1
        else:
            self.troops[new_square] += moved_troops
            self.owns[new_square] = self.player

        self.troops[square] = 1

    def move(self, square, direction):
        """Update the game state.

        square - tuple indicating position to move from
        direction - letter from {u, d, l, r} indicating which direction to move

        Assumes this is a valid move.
        """
        self._make_move(square, direction)
        self._update_game()
