import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import sys

class ArmsEnv(gym.Env):
    """
    Two player game.
    There are N (say 4) possible arms to pull.
        0 1 2 3
    Some of the arms are forbidden (determined by the state)
    Goal is to pull same arm.
    """

    def __init__(self, n, m):
        super(ArmsEnv, self).__init__()
        self.n = n
        self.m = m          # number of contexts with hard rules
        self.action_space = spaces.MultiDiscrete([2*n, 2*n])
        self.observation_space = spaces.MultiDiscrete([n])
        self.reset()

        self.invert = False

    def step(self, a):
        context = self.state[0]
        match = (a[0] == a[1])

        if context < self.m:
            if not self.invert: 
                correct = match and a[0] == context
            else:
                correct = match and a[0] - self.n == context
        else:
            correct = match and a[0]%self.n == context                              # action mod n equals context
        
        self.reward = (int)(correct)
        return [self.state, self.reward, True, {}]

    def reset(self, state=None):
        if not hasattr(self, 'last'): self.last = 0

        self.rep = 0
        if state: 
            self.state = state
        else:
            self.state = [np.random.randint(self.n)]

        return self.state

    def render(self):
        print(self.n, self.state)

    def set_invert(self, invert):
        self.invert = invert