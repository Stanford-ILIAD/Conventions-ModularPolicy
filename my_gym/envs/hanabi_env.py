import gym
from gym import error, spaces, utils
from hanabi_learning_environment.rl_env import HanabiEnv

class HanabiEnvWrapper(HanabiEnv, gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, config):
        self.config = config
        super(HanabiEnvWrapper, self).__init__(config=self.config)

        observation_shape = super().vectorized_observation_shape()
        self.observation_space = spaces.MultiBinary(observation_shape[0])
        self.action_space = spaces.Discrete(self.game.max_moves())
    
    def reset(self):
        obs = super().reset()
        obs = obs['player_observations'][obs['current_player']]['vectorized']
        return obs

    def step(self, action):
        # action is a integer from 0 to self.action_space
        # we map it to one of the legal moves
        # the legal move array may be too small in some cases, so just modulo action by the array length
        legal_moves = self.state.legal_moves()
        move = legal_moves[action % len(legal_moves)].to_dict()

        obs, reward, done, info = super().step(move)
        obs = obs['player_observations'][obs['current_player']]['vectorized']
        return obs, reward, done, info