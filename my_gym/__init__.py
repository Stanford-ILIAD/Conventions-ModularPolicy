from gym.envs.registration import register

register(
    id='blocks-v0',
    entry_point='my_gym.envs:BlocksEnv',
)

register(
    id='arms-v0',
    entry_point='my_gym.envs:ArmsEnv',
)

register(
    id='hanabi-v0',
    entry_point='my_gym.envs:HanabiEnvWrapper',
)

register(
    id='arms-human-v0',
    entry_point='my_gym.envs:ArmsHumanEnv',
)