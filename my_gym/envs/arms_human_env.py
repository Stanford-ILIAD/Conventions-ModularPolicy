import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import sys

class ArmsHumanEnv(gym.Env):
    """
    Two player game.
    There are N (say 4) possible arms to pull.
        0 1 2 3
    Some of the arms are forbidden (determined by the state)
    Goal is to pull same arm.
    """

    def __init__(self):
        super(ArmsHumanEnv, self).__init__()
        self.n = 3
        self.a = 4
        self.action_space = spaces.MultiDiscrete([self.a, self.a])
        self.observation_space = spaces.MultiDiscrete([self.n])
        self.reset()

    def step(self, a):
        context = self.state[0]
        match = (a[0] == a[1])
        green = [
            [1,0,0,0],
            [0,0,1,1],
            [0,1,0,1],
        ]
        correct = match and green[context][a[0]]
        self.reward = (int)(correct)
        return [self.state, self.reward, True, {}]

    def reset(self, state=None):
        if state: 
            self.state = state
        else:
            self.state = [np.random.randint(self.n)]

        return self.state

    def render(self):
        print(self.n, self.state)