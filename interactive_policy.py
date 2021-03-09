from abc import ABC, abstractmethod
import collections
from typing import Union, Type, Dict, List, Tuple, Optional, Any, Callable

import torch as th
import torch.nn as nn
import numpy as np

from partner import Partner, PartnerPolicy, PPOPartnerPolicy
from modular_policy import ModularPolicy

class InteractivePolicy(ModularPolicy):
    def __init__(self, *args, **kwargs):
        super(InteractivePolicy, self).__init__(*args, **kwargs)
        self.partners = None

    def set_partners(self, partners: Optional[List[Partner]]=None):
        self.partners = partners
        self.num_partners = len(partners) if partners is not None else 1
        
    def set_PPO_partners(self, partner_model_paths: List[str]):
        self.set_partners(partners=[Partner(PPOPartnerPolicy(pmpath)) for pmpath in partner_model_paths])

class OptimalPolicy(ModularPolicy, ABC):
    def __init__(self, *args, **kwargs):
        super(OptimalPolicy, self).__init__(*args, **kwargs)
        self.use_optimal_mask = False

    @abstractmethod
    def get_mask(self, obs):
        pass

    @abstractmethod
    def setup_optimal_mask(self, env):
        pass

    def evaluate_actions(self, obs: th.Tensor,
                         actions: th.Tensor,
                         partner_idx: int) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        optimal_mask = self.get_mask(obs) if self.use_optimal_mask else None
        return super(OptimalPolicy, self).evaluate_actions(obs=obs, actions=actions, partner_idx=partner_idx, action_mask=optimal_mask)

    def get_action_dist_from_obs(self, obs: th.Tensor, partner_idx: int) -> th.Tensor:
        optimal_mask = self.get_mask(obs) if self.use_optimal_mask else None
        return super(OptimalPolicy, self).get_action_dist_from_obs(obs=obs, partner_idx=partner_idx, action_mask=optimal_mask)

class BlocksPolicy(InteractivePolicy, OptimalPolicy):
    def __init__(self, *args, **kwargs):
        super(BlocksPolicy, self).__init__(*args, **kwargs)

    def setup_optimal_mask(self, env):
        self.use_optimal_mask = True
        self.sz = env.grid_size ** 2
        self.action_sz = env.action_space.n

    def get_mask(self, obs):
        optimal_mask = th.ones((obs.size(0), self.action_sz), dtype=th.bool)
        
        # 6 turns so turn 0 is the first move for P1. Target is blue block which has id=3.
        first_move, second_move, target_id = 0, 2, 3
        is_first_two_moves = th.logical_or(obs[:,2*self.sz] == first_move, obs[:,2*self.sz] == second_move)
        goal_obs = obs[:,:self.sz]

        optimal_mask[is_first_two_moves] = th.zeros(self.action_sz, dtype=th.bool)     
        loc_of_target = (goal_obs[is_first_two_moves] != target_id)
        optimal_mask[is_first_two_moves, :self.sz] = loc_of_target

        # 6 turns so turn 4 is the last move for P1. Target is red block which has id=2.
        last_move, target_id = 4, 2
        is_last_move = (obs[:,2*self.sz] == last_move)
        goal_obs = obs[:,:self.sz]

        # this part only works nicely since the goal obs exactly matches optimal action representation
        optimal_mask[is_last_move] = th.zeros(self.action_sz, dtype=th.bool)     # if last move, all moves are suboptimal except one
        loc_of_target = (goal_obs[is_last_move] == target_id) 
        optimal_mask[is_last_move, :self.sz] = loc_of_target




        return optimal_mask

    def forward(self, obs: th.Tensor,
                partner_idx: int,
                deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)
        :param obs: (th.Tensor) Observation
        :param deterministic: (bool) Whether to sample or use deterministic actions
        :return: (Tuple[th.Tensor, th.Tensor, th.Tensor]) action, value and log probability of the action
        """
        if self.is_partners_turn(obs) and self.partners is not None:
            actions, values, log_probs = self.partners[partner_idx].policy.forward(obs=obs, deterministic=deterministic)
            return actions, th.tensor([0.0]), th.tensor([0.0]) # effectively detaching value / log_prob of partner

        latent_pi, latent_vf, _ = self._get_latent(obs=obs)
        partner_latent_pi, partner_latent_vf = self.partner_mlp_extractor[partner_idx](latent_pi)

        distribution = self._get_action_dist_from_latent(latent_pi, partner_latent_pi, partner_idx=partner_idx)
        if self.use_optimal_mask: # limit actions to optimal actions
            optimal_mask = self.get_mask(obs)
            distribution = self._get_action_dist_from_latent(latent_pi, partner_latent_pi, partner_idx=partner_idx, action_mask=optimal_mask)

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf) + self.partner_value_net[partner_idx](partner_latent_vf)

        return actions, values, log_prob

    def is_partners_turn(self, obs):
        turn = obs[0][-1]
        return turn % 2 == 1


class ArmsPolicy(InteractivePolicy, OptimalPolicy):
    def __init__(self, *args, **kwargs):
        super(ArmsPolicy, self).__init__(*args, **kwargs)

    def setup_optimal_mask(self, env):
        self.use_optimal_mask = True
        from tabular import tabular_q_learning
        self.q_values, self.optimal_action1_mask, self.optimal_action2_mask = tabular_q_learning(env)

    def get_mask(self, obs):
        obs_idx = tuple(obs.T.long())
        optimal_mask = th.cat((self.optimal_action1_mask[obs_idx], self.optimal_action2_mask[obs_idx]), dim=1)
        return optimal_mask

    def forward(self, obs: th.Tensor,
                partner_idx: int,
                deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)
        :param obs: (th.Tensor) Observation
        :param deterministic: (bool) Whether to sample or use deterministic actions
        :return: (Tuple[th.Tensor, th.Tensor, th.Tensor]) action, value and log probability of the action
        """
        latent_pi, latent_vf, _ = self._get_latent(obs=obs)
        partner_latent_pi, partner_latent_vf = self.partner_mlp_extractor[partner_idx](latent_pi)

        distribution = self._get_action_dist_from_latent(latent_pi, partner_latent_pi, partner_idx=partner_idx)
        if self.use_optimal_mask: # limit actions to optimal actions
            optimal_mask = self.get_mask(obs)
            distribution = self._get_action_dist_from_latent(latent_pi, partner_latent_pi, partner_idx=partner_idx, action_mask=optimal_mask)

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf) + self.partner_value_net[partner_idx](partner_latent_vf)

        # partner actions
        if self.partners is not None:
            partner_actions, _, _ = self.partners[partner_idx].policy.forward(obs=obs, deterministic=deterministic)
            partner_actions = partner_actions.to(actions.device)
            actions = th.stack((actions[:,0], partner_actions[:,1]), dim=1)

        #print(obs, actions, log_prob)
        return actions, values, log_prob

class HanabiPolicy(InteractivePolicy):
    def __init__(self, *args, **kwargs):
        super(HanabiPolicy, self).__init__(*args, **kwargs)
        self.turn = 0

    def forward(self, obs: th.Tensor,
                partner_idx: int,
                deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)
        :param obs: (th.Tensor) Observation
        :param deterministic: (bool) Whether to sample or use deterministic actions
        :return: (Tuple[th.Tensor, th.Tensor, th.Tensor]) action, value and log probability of the action
        """
        self.turn = 1 - self.turn
        if self.is_partners_turn(obs) and self.partners is not None:
            actions, values, log_probs = self.partners[partner_idx].policy.forward(obs=obs, deterministic=deterministic)
            return actions, th.tensor([0.0]), th.tensor([0.0]) # effectively detaching value / log_prob of partner

        latent_pi, latent_vf, _ = self._get_latent(obs=obs)
        partner_latent_pi, partner_latent_vf = self.partner_mlp_extractor[partner_idx](latent_pi)

        distribution = self._get_action_dist_from_latent(latent_pi, partner_latent_pi, partner_idx=partner_idx)

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf) + self.partner_value_net[partner_idx](partner_latent_vf)

        return actions, values, log_prob

    def is_partners_turn(self, obs):
        return self.turn == 1