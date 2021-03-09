import warnings
import argparse
import os, time
import sys

import gym
import my_gym
from hanabi_learning_environment import pyhanabi

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import torch as th
import torch.nn as nn
from interactive_policy import HanabiPolicy
from partner_config import get_hanabi_partners
from util import check_optimal, learn, load_model
from util import adapt_task, adapt_partner_baseline, adapt_partner_modular, adapt_partner_scratch

warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--n',              type=int, default=4,     help="n arms")
parser.add_argument('--run',            type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--netsz',          type=int, default=500,     help="Size of policy network")
parser.add_argument('--latentz',        type=int, default=500,    help="Size of latent z dimension")

parser.add_argument('--mreg',           type=float, default=0.0, help="Marginal regularization.")
parser.add_argument('--baseline',       action='store_true', default=False, help="Baseline: no modular separation.")
parser.add_argument('--nomain',         action='store_true', default=False, help="Baseline: don't use main logits.")

parser.add_argument('--timesteps',      type=int, default=500000,     help="Number of timesteps to train for")
parser.add_argument('--selfplay',       action='store_true', default=False, help="converge using selfplay")
parser.add_argument('--testing',        action='store_true', default=False, help="Testing.")
parser.add_argument('--zeroshot',       action='store_true', default=False, help="Try zeroshot combination of task + partner.")

parser.add_argument('--k',              type=int, default=0, help="When fixedpartner=True, k is the index of the test partner")

parser.add_argument('--colors',                  type=int, default=1,     help="number of card colors in the game")
parser.add_argument('--ranks',                   type=int, default=5,     help="number of card ranks in the game")
parser.add_argument('--hand_sz',                    type=int, default=2,     help="hand size of each player")
parser.add_argument('--info',              type=int, default=3,     help="number of information tokens")
parser.add_argument('--life',              type=int, default=3,     help="number of life tokens")

args = parser.parse_args()
print(args)

def get_model_name_and_path(run, mreg=0.00):
    layout = [
        ('n={:01d}', args.n),
        ('run={:04d}', run),
        ('netsz={:03d}',  args.netsz),
        ('mreg={:.2f}', mreg),
    ]

    m_name = '_'.join([t.format(v) for (t, v) in layout])
    m_path = 'output/hanabi_' + m_name
    return m_name, m_path

model_name, model_path = get_model_name_and_path(args.run, mreg=args.mreg)

HP = {
    'n_steps': 640,
    'n_steps_testing': 640,
    'batch_size': 160, 
    'n_epochs': 5,
    'n_epochs_testing': 5,
    'mreg': args.mreg,
}

config = {
    "colors":                   args.colors,
    "ranks":                    args.ranks,
    "players":                  2,
    "hand_size":                args.hand_sz,
    "max_information_tokens":   args.info,
    "max_life_tokens":          args.life,
    "observation_type":         pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
}

env = gym.make('hanabi-v0', config=config)

if args.selfplay:
    PARTNERS = None     # this must be None to trigger selfplay
else:
    setting, partner_type = "", "ppo"
    TRAIN_PARTNERS, TEST_PARTNERS = get_hanabi_partners(setting, partner_type)
    PARTNERS = [ TEST_PARTNERS[args.k % len(TEST_PARTNERS)] ] if args.testing else TRAIN_PARTNERS

def main():
    global PARTNERS
    num_partners = len(PARTNERS) if PARTNERS is not None else 1

    print("model path: ", model_path)
    net_arch = [args.netsz,args.latentz]
    partner_net_arch = [args.netsz,args.netsz]
    policy_kwargs = dict(activation_fn=nn.ReLU,
                         net_arch=[dict(vf=net_arch, pi=net_arch)],
                         partner_net_arch=[dict(vf=partner_net_arch, pi=partner_net_arch)],
                         num_partners=num_partners,
                         baseline=args.baseline,
                         nomain=args.nomain,
                         )

    def load_model_fn(partners, testing, try_load=True):
        return load_model(model_path=model_path, policy_class=HanabiPolicy, policy_kwargs=policy_kwargs, env=env, hp=HP, partners=partners, testing=testing, try_load=try_load)

    def learn_model_fn(model, timesteps, save, period):
        return learn(model, model_name=model_name, model_path=model_path, timesteps=timesteps, save=save, period=period)

    # TRAINING
    if not args.testing:
        print("#section Training")
        model = load_model_fn(partners=PARTNERS, testing=False)
        learn_model_fn(model, timesteps=args.timesteps, save=True, period=2000)

    ts, period = 25600, HP['n_steps_testing']
    # TESTING
    if args.testing and not args.zeroshot:
        if args.baseline:   adapt_partner_baseline(load_model_fn, learn_model_fn, partners=PARTNERS, timesteps=ts, period=period, do_optimal=False)
        else:               adapt_partner_modular(load_model_fn, learn_model_fn, partners=PARTNERS, timesteps=ts, period=period, do_optimal=False)
        adapt_partner_scratch(load_model_fn, learn_model_fn, partners=PARTNERS, timesteps=ts, period=period, do_optimal=False)

    if args.testing and args.zeroshot:
        adapt_task(load_model_fn, learn_model_fn, train_partners=TRAIN_PARTNERS, test_partners=TEST_PARTNERS, timesteps=args.timesteps, period=200)

if __name__ == "__main__":
    main()