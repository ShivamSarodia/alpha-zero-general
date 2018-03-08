import copy
import numpy as np
import sys

sys.path.append('..')
from Game import Game

class GeneralsBoard:
    """Board of a Generals game.

    width (int) - width of the map
    height (int) - height of the map
    step (int) - current step of the game. steps alternate between P1
                 and P2, and a turn is every 4 steps

    troops (array) - array of troop numbers. a positive number means
                     player +1 has troops there, and a negative number
                     means player -1 has troops there. 0 means neither
                     player has troops there, but the square may still
                     be owned by a player.

    owns (array) - array of +1 if player 1 owns a square, -1 if player
                   -1 owns it, and 0 if neither owns it

    generals (array) - array of +1 at player +1's general and -1 at
                       player -1's general

    TODO: add mountains and cities

    """

    def __init__(self, width, height):
        # step number (step / 2 = turn)
        # troops: how many troops on each square
        # who owns: + or -
        # features: mountains, city, general, etc.

        self.width = width
        self.height = height
        self.step = 0
        self.troops = np.zeros((width, height))
        self.owns = np.zeros((width, height))
        self.generals = np.zeros((width, height))

        # Initialize
        self.generals[0,0] = 1
        self.owns[0,0] = 1
        self.generals[width-1, height-1] = -1
        self.owns[width-1, height-1] = -1

        # Give all generals one troop
        self.troops += self.generals

    def copy(self):
        g = GeneralsBoard(self.width, self.height)
        g.step = self.step
        g.troops = np.copy(self.troops)
        g.owns = np.copy(self.owns)
        g.generals = np.copy(self.generals)
        return g

    def _update_game(self, player):
        # only increment step after player -1's turn
        self._grow_troops(player)
        self.step += 1

    def _grow_troops(self, player):
        # Don't grow on step 0 or on odd steps, and only grow for
        # current player.
        if self.step > 1 and self.step % 50 in {0, 1}:
            self.troops += player * (self.owns == player)
        elif self.step > 1 and self.step % 4 in {0, 1}:
            self.troops += player * (self.generals == player)

    def _make_move(self, square, direction, player):
        if self.owns[square] != player:
            return

        if direction == 0: # up
            new_square = (square[0] - 1, square[1])
        elif direction == 1: # left
            new_square = (square[0], square[1] - 1)
        elif direction == 2: # down
            new_square = (square[0] + 1, square[1])
        elif direction == 3: # right
            new_square = (square[0], square[1] + 1)
        else:
            raise ValueError

        if not 0 <= new_square[0] < self.height:
            return
        elif not 0 <= new_square[1] < self.width:
            return

        moved_troops = self.troops[square] - player
        if moved_troops * player <= 0:
            return

        self.troops[new_square] += moved_troops
        if self.troops[new_square] < 0:
            self.owns[new_square] = -1
        elif self.troops[new_square] > 0:
            self.owns[new_square] = 1

        self.troops[square] = player

    def move(self, square, direction, player):
        """Update the game state.

        square - tuple indicating position to move from
        direction - letter from {u, d, l, r} indicating which direction to move

        Assumes this is a valid move.
        """
        self._make_move(square, direction, player)
        self._update_game(player)

    def valid_moves(self, player):
        """Return a numpy array with a 1 everywhere owned by given player."""
        return (self.owns == player) & ((self.troops * player > 1) | (self.generals == player))

    def game_ended(self, player):
        if self.owns[self.generals == -player] == player:
            return 1
        elif self.owns[self.generals == player] == -player:
            return -1
        elif self.step > 100 and np.sum(self.troops * player) > 0:
            return 1
        elif self.step > 100 and np.sum(self.troops * player) < 0:
            return -1
        elif self.step > 100:
            return 1e-4
        else:
            return 0

    def flip_form(self):
        self.troops *= -1
        self.owns *= -1
        self.generals *= -1

    def to_string(self):
        return (str(self.step) +
                str(self.troops.tostring()) +
                str(self.owns.tostring()) +
                str(self.generals.tostring()))

    def to_array(self):
        shape_layer = np.full(self.troops.shape, self.step % 50)
        return np.concatenate([shape_layer.flatten(),
                               self.troops.flatten(),
                               self.owns.flatten(),
                               self.generals.flatten()])

    def show(self, player):
        print("Step {}, To Move {}:".format(self.step, player))
        print(self.troops)

    def hflip(self):
        self.troops = np.fliplr(self.troops)
        self.owns = np.fliplr(self.owns)
        self.generals = np.fliplr(self.generals)

    def vflip(self):
        self.troops = np.flipud(self.troops)
        self.owns = np.flipud(self.owns)
        self.generals = np.flipud(self.generals)


class GeneralsGame(Game):
    """Computes Generals game features."""

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def getInitBoard(self):
        return GeneralsBoard(self.width, self.height)

    def getBoardSize(self):
        return self.width, self.height

    def getActionSize(self):
        return self.width * self.height * 4

    def action_to_move(self, action):
        direction = int(action / (self.width * self.height))
        action = action % (self.width * self.height)

        y = int(action / self.width)
        x = action % self.width

        return y, x, direction

    def getNextState(self, board, player, action):
        next_board = board.copy()
        y, x, direction = self.action_to_move(action)

        next_board.move((y,x), direction, player)
        return next_board, -player

    def getValidMoves(self, board, player):
        valid_moves = board.valid_moves(player)
        valid_flat = valid_moves.flatten()
        return np.tile(valid_flat, 4)

    def getGameEnded(self, board, player):
        return board.game_ended(player)

    def getCanonicalForm(self, board, player):
        if player == 1:
            return board
        else:
            board = board.copy()
            board.flip_form()
            return board

    def hflip_pi(self, pi):
        shape_pi = np.array(pi).reshape(4, self.height, self.width)
        flipped_pi = np.flip(shape_pi, axis=2)
        pref_left = np.copy(flipped_pi[1])
        flipped_pi[1] = flipped_pi[3]
        flipped_pi[3] = pref_left
        return flipped_pi.flatten()

    def vflip_pi(self, pi):
        shape_pi = np.array(pi).reshape(4, self.height, self.width)
        flipped_pi = np.flip(shape_pi, axis=1)
        pref_up = np.copy(flipped_pi[0])
        flipped_pi[0] = flipped_pi[2]
        flipped_pi[2] = pref_up
        return flipped_pi.flatten()

    def getSymmetries(self, board, pi):
        symmetries = [(board, pi)]

        hflip_pi = self.hflip_pi(pi)
        hflip_board = board.copy()
        hflip_board.hflip()
        symmetries.append((hflip_board, hflip_pi))

        vflip_pi = self.vflip_pi(pi)
        vflip_board = board.copy()
        vflip_board.vflip()
        symmetries.append((vflip_board, vflip_pi))

        hvflip_pi = self.vflip_pi(hflip_pi)
        hvflip_board = hflip_board.copy()
        hvflip_board.vflip()
        symmetries.append((hvflip_board, hvflip_pi))

        return symmetries

    def stringRepresentation(self, board):
        return board.to_string()

# g = GeneralsGame(5,5)
# board = g.getInitBoard()
# player = 1

# while True:
#     board.show(player)
#     x = int(input("x (from left): "))
#     y = int(input("y (from top): "))
#     d = int(input("direction: "))
#     board, player = g.getNextState(board, player, 0)
