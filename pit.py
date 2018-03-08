import Arena
from MCTS import MCTS
from generals.GeneralsGame import GeneralsGame
from generals.GeneralsNN import NNetWrapper as NNet

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a

def display(board):
    board.show("?")

g = GeneralsGame(5, 5)

# all players
rp = RandomPlayer(g).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint('./temp/','checkpoint_18.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

arena = Arena.Arena(n1p, rp, g, display=display)
print(arena.playGames(2, verbose=True))
