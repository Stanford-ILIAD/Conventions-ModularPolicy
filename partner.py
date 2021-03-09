from abc import ABC, abstractmethod
from typing import Tuple
import torch as th
import numpy as np
from stable_baselines3 import PPO

class PartnerPolicy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, obs, deterministic=True) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        pass

class Partner:
    def __init__(self, policy : PartnerPolicy):
        self.policy = policy

class PPOPartnerPolicy(PartnerPolicy):
    def __init__(self, model_path):
        super(PartnerPolicy, self).__init__()
        self.model = PPO.load(model_path)
        print("PPO Partner loaded successfully: %s" % model_path)

    def forward(self, obs, deterministic=True):
        return self.model.policy.forward(obs, partner_idx=0, deterministic=deterministic)

class BlocksPermutationPartnerPolicy(PartnerPolicy):
    def __init__(self, perm, n=2):
        super(PartnerPolicy, self).__init__()
        self.perm = perm
        self.n = n

        self.action_index = [[i*n + j for j in range(n)] for i in range(n)]

    def forward(self, obs, deterministic=True):
        obs = obs[0]
        assert(2*self.n**2+1 == len(obs))
        goal_grid = obs[:self.n**2].reshape(self.n,self.n)
        working_grid = obs[self.n**2:2*(self.n**2)].reshape(self.n,self.n)
        turn = obs[-1]

        r, c = self.get_red_block_position(working_grid, self.n, self.n)
        
        #if r == None or turn >= 2:
        if r == None or turn >= 2:
            action = self.n**2+1        # pass turn
        else:
            action = self.perm[self.action_index[r][c]]

        return th.tensor([action]), th.tensor([0.0]), th.tensor([0.0])

    def get_block_position(self, grid, r, c, target):
        for i in range(r):
            for j in range(c):
                if grid[i][j] == target:
                    return i, j
        return None, None

    def get_blue_block_position(self, grid, r, c):
        return self.get_block_position(grid, r, c, 3)

    def get_red_block_position(self, grid, r, c):
        return self.get_block_position(grid, r, c, 2)

class ArmsPartnerPolicy(PartnerPolicy):
    def __init__(self, perm):
        super(PartnerPolicy, self).__init__()
        self.perm = th.tensor(perm)

    def forward(self, obs, deterministic=True):
        action = self.perm[obs]
        return th.cat((action, action), dim=1), th.tensor([0.0]), th.tensor([0.0])

class LowRankPartnerPolicy(PartnerPolicy):
    def __init__(self, n):
        super(PartnerPolicy, self).__init__()
        self.n = n

    def forward(self, obs, deterministic=True):
        action = self.n
        return th.tensor([action]), th.tensor([0.0]), th.tensor([0.0])