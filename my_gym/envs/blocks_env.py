import gym
from gym import error, spaces, utils
from gym.utils import seeding

from my_gym.envs.generate_grid import make_grids

import numpy as np
import sys


BLOCKS = [ # per owner
    # height, width
    (1, 1),
]

BLOCKS_PER_COLOR_ON_GRID = 1
SNAPSHOT_CNT = 2

EMPTY_COLOR = 0

HISTORY = False

class Block:
    def __init__(self, owner, color, height, width, row, col):
        self.owner, self.color = owner, color
        self.height, self.width = height, width
        self.row, self.col = row, col
    def get_location(self):
        return self.row, self.col
    def set_location(self, nr, nc):
        self.row, self.col = nr, nc
    def get_owner(self):
        return self.owner
    def get_color(self):
        return self.color
    def get_size(self):
        return self.height, self.width

class GameState:
    def __init__(self, nrow, ncol, goal_grid, p1_grid, p2_grid, blocks, max_move_number):
        self.nrow, self.ncol = nrow, ncol
        self.goal_grid = goal_grid
        self.p1_grid = p1_grid
        self.p2_grid = p2_grid
        self.blocks = blocks
        self.nblocks = len(blocks)
        self.blocks_history = [[0 for _ in self.blocks] for _ in range(max_move_number)]
        self.move_number = 0
        self.max_move_number = max_move_number

    def get_move_number(self):
        return self.move_number
    def set_move_number(self, m):
        self.move_number = m
    def get_player_turn(self):
        return (self.get_move_number() % 2) + 1

    ### HISTORY
    def convert_location_to_number(self, r, c):
        if r == -1: return 1
        return 2 + r * self.ncol + c
    def add_to_history(self, move_number):
        snapshot = [b.get_location() for b in self.blocks]
        snapshot = [self.convert_location_to_number(r, c) for r, c in snapshot]
        self.blocks_history[move_number] = snapshot
    ###

    def construct_working_grid(self):
        grid = [[0 for _ in range(self.ncol)] for _ in range(self.nrow)]
        for i in range(self.nblocks):
            color = self.blocks[i].get_color()
            r, c = self.blocks[i].get_location()
            h, w = self.blocks[i].get_size()

            if r != -1:
                for ii in range(r, r+h):
                    for jj in range(c, c+w):
                        grid[ii][jj] = color
        return grid

    def serialize(self, p=None):
        if p is None:
            p = self.get_player_turn()
        grid = self.p1_grid if p == 1 else self.p2_grid

        move_number = self.get_move_number()
        vis_grid_flat = [x for row in grid for x in row]
        work_grid_flat = [x for row in self.construct_working_grid() for x in row]
        
        if HISTORY:
            history = self.blocks_history[max(0,move_number-SNAPSHOT_CNT):move_number] # take latest SNAPSHOT_CNT snapshots
            history = [[1 for _ in self.blocks] for _ in range(SNAPSHOT_CNT-move_number)] + history # pad with empty snapshots if not enough history
            blocks_flat = [x for snapshot in history for x in snapshot]
            return np.array(vis_grid_flat + work_grid_flat + blocks_flat + [move_number]) # make sure last value is move number, used in custom_net_utils

        return np.array(vis_grid_flat + work_grid_flat + [move_number]) # make sure last value is move number, used in custom_net_utils

class BlocksEnv(gym.Env):
    """
    A grid like the following

        0000
        2230
        0030
        0000

    0 : empty grid
    1 : hidden grid
    2 : player 1 (red)
    3 : player 2 (blue)

    """
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size, vis1, vis2, one_sided_reward=False, max_move_number=6):
        # # number of actions:
        # # for each block we can move it onto any square or into inventory
        # # or we can pass or end game
        self.grid_size = grid_size
        self.vis1 = vis1
        self.vis2 = vis2
        self.one_sided_reward = one_sided_reward
        self.invert = False

        self.nrow, self.ncol = grid_size, grid_size
        self.nblocks = 2*len(BLOCKS)
        self.max_move_number = max_move_number
        self.nA = len(BLOCKS) * (1 + self.nrow * self.ncol) + 1 # +1 for pass

        # # number of states:
        # # visible grid - nrow * ncol
        # # working grid - row and col of each block
        # # history - (working grid) * (length of game)
        # nS = 2*nblocks                 # location of blocks
        #     + (2*nblocks*nmoves)        # history of location of blocks
        #     + nrow * ncol               # visible grid
        #     + move_num                  # number of moves
        super(BlocksEnv, self).__init__()
        self.action_space = spaces.Discrete(self.nA)

        self.observation_space = spaces.MultiDiscrete(
            [4 for _ in range(self.nrow * self.ncol)] + 
            [4 for _ in range(self.nrow * self.ncol)] + 
            ([self.nrow * self.ncol + 2 for _ in range(self.nblocks * SNAPSHOT_CNT)] if HISTORY else []) + 
            [self.max_move_number+1]
        )

        self.reset()

    def valid_move(self, i, nr, nc):
        # Check if moving the i-th block to (nr, nc) is a valid move
        # Note: if the i-th block is already at (nr, nc), then the move counts as invalid
        assert(i >= 0 and i < self.state.nblocks)
        player = self.state.get_player_turn()
        if self.state.blocks[i].get_owner() != player:
            return False # not moving our own block

        r, c = self.state.blocks[i].get_location()
        h, w = self.state.blocks[i].get_size()
        if not (nr == -1 and nc == -1) and not (nr >= 0 and nr+h <= self.nrow and nc >= 0 and nc+w <= self.ncol):
            return False

        if r == nr and c == nc: # if block is already in position
            return False
        if nr == -1: # if block is in inventory, then we can move it anywhere
            return True

        # check that new space is not occupied by any other block
        r1, r2, c1, c2 = nr, nr+h, nc, nc+w
        for j in range(self.state.nblocks):
            if i == j: continue
            nro, nco = self.state.blocks[j].get_location()
            ho, wo = self.state.blocks[j].get_size()
            r1o, r2o, c1o, c2o = nro, nro+ho, nco, nco+wo
            if nro == -1: continue
            if r1 >= r2o or r1o >= r2 or c1 >= c2o or c1o >= c2: continue

            # space is occupied, remove existing block
            self.state.blocks[j].set_location(-1,-1)
        return True

    def make_move(self, i, nr, nc):
        self.state.blocks[i].set_location(nr, nc)
        self.state.add_to_history(self.state.get_move_number())
        self.state.set_move_number(self.state.get_move_number()+1)

    def do_pass(self):
        self.state.add_to_history(self.state.get_move_number())
        self.state.set_move_number(self.state.get_move_number()+1)

    def do_endgame(self):
        self.done = True
        self.reward = self.calculate_reward()

    def calculate_reward(self):
        reward = 0

        grid = np.array(self.state.construct_working_grid())

        if self.invert:
            # swap the 2 and the 0 entries
            # red block has to be placed in one of the empty blocks
            grid[grid == 2] = -1
            grid[grid == 0] = 2
            grid[grid == -1] = 0

        for r in range(self.nrow):
            for c in range(self.ncol):
                if self.one_sided_reward:
                    if grid[r][c] == 3:
                        reward += 20 * (int)(grid[r][c] == self.state.goal_grid[r][c])
                else:
                    if grid[r][c] != EMPTY_COLOR:
                        reward += 10 * (int)(grid[r][c] == self.state.goal_grid[r][c])

        # for r in range(self.nrow):
        #     print(str(grid[r]))
        # print(self.state.goal_grid)
        # print(self.state.p1_grid)
        # print(self.state.p2_grid)
        # print(reward, "\n")
        return reward

    def do_action(self, a):
        '''
            0 to r*c-1  : place block in position
            r*c         : remove block
            r*c + 1     : pass
        '''
        must_end = self.state.get_move_number() == self.state.max_move_number
        if must_end:
            self.do_endgame()
        elif a == self.nA-1:
            self.do_pass()
        else:
            i = a // (1 + self.nrow * self.ncol)
            a %= (1 + self.nrow * self.ncol)
            nr, nc = a // self.ncol, a % self.ncol
            if a == self.nrow * self.ncol:
                nr, nc = -1, -1

            # get the i-th block of the current player
            player = self.state.get_player_turn()
            if player == 2:
                i += self.nblocks // 2
            assert(self.state.blocks[i].get_owner() == player)

            if not self.valid_move(i, nr, nc):
                self.do_pass()
                #self.reward = -1e3
            else:
                self.make_move(i, nr, nc)
                #self.reward = self.calculate_reward()


    def step(self, a):
        # print(self.state.p1_grid)
        # print(self.state.p2_grid)
        # print(self.state.construct_working_grid())
        self.do_action(a)
        return [self.state.serialize(), self.reward, self.done, {}]

    def do_reset(self, verbose=False, rep=0):
        goal_grid, p1_grid, p2_grid = make_grids(self.grid_size, self.vis1, self.vis2, BLOCKS, BLOCKS_PER_COLOR_ON_GRID)

        blocks = [Block(1, 2, x[0], x[1], -1, -1) for x in BLOCKS] + [Block(2, 3, x[0], x[1], -1, -1) for x in BLOCKS]
        self.state = GameState(self.nrow, self.ncol, goal_grid, p1_grid, p2_grid, blocks, self.max_move_number)
        self.reward = 0
        self.done = False
        self.rep = rep

        if verbose:
            print(self.state.goal_grid)
            print(self.state.p1_grid)
            print(self.state.p2_grid)

    def reset(self, verbose=0):
        self.do_reset(verbose)
        return self.state.serialize()

    def render(self, mode='human', close=False, verbose=0):
        if close:
            return
        out = sys.stdout

        if verbose:
            grid = self.state.construct_working_grid()
            for r in range(self.nrow):
                out.write(str(grid[r]) + "\n")
            out.write("Move: %u, Reward: %u\n" % (self.state.get_move_number(), self.reward))

        # calculate max possible reward
        max_possible_reward = 0
        for r in range(self.state.nrow):
            for c in range(self.state.ncol):
                max_possible_reward += 10 * int(self.state.p1_grid[r][c] == 2 or self.state.p1_grid[r][c] == 3
                                        or self.state.p2_grid[r][c] == 2 or self.state.p2_grid[r][c] == 3)

        return max_possible_reward


    def set_invert(self, invert):
        self.invert = True

    def switch_to_env(self, idx):
        pass