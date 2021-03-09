import warnings
import argparse
import os, time

import gym
import my_gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import torch as th
import torch.nn as nn
from interactive_policy import ArmsPolicy
from partner_config import get_arms_human_partners
from util import check_optimal, learn, load_model
from util import adapt_task, adapt_partner_baseline, adapt_partner_modular, adapt_partner_scratch

warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--run',            type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--netsz',          type=int, default=30,    help="Size of policy network")
parser.add_argument('--latentz',        type=int, default=30,    help="Size of latent z dimension")

parser.add_argument('--mreg',          type=float, default=0.0, help="Marginal regularization.")
parser.add_argument('--baseline',       action='store_true', default=False, help="Baseline: no modular separation.")
parser.add_argument('--nomain',       action='store_true', default=False, help="Baseline: don't use main logits.")

parser.add_argument('--timesteps',      type=int, default=10000,     help="Number of timesteps to train for")
parser.add_argument('--testing',        action='store_true', default=False, help="Testing.")

parser.add_argument('--k',              type=int, default=0, help="When fixedpartner=True, k is the index of the test partner")

args = parser.parse_args()
print(args)

def get_model_name_and_path(run, mreg=0.00):
    layout = [
        ('run={:04d}', run),
        ('netsz={:03d}',  args.netsz),
        ('mreg={:.2f}', mreg),
    ]

    m_name = '_'.join([t.format(v) for (t, v) in layout])
    m_path = 'output/armshuman_' + m_name
    return m_name, m_path

model_name, model_path = get_model_name_and_path(args.run, mreg=args.mreg)

HP = {
    'n_steps': 64,
    'n_steps_testing': 16,
    'batch_size': 16,
    'n_epochs': 20,
    'n_epochs_testing': 50,
    'mreg': args.mreg,
}

setting, partner_type = "", "fixed"
TRAIN_PARTNERS, TEST_PARTNERS = get_arms_human_partners(setting, partner_type)
PARTNERS = [ TEST_PARTNERS[args.k % len(TEST_PARTNERS)] ] if args.testing else TRAIN_PARTNERS

def main():
    global PARTNERS
    env = gym.make('arms-human-v0')
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
        return load_model(model_path=model_path, policy_class=ArmsPolicy, policy_kwargs=policy_kwargs, env=env, hp=HP, partners=partners, testing=testing, try_load=try_load)

    def learn_model_fn(model, timesteps, save, period):
        return learn(model, model_name=model_name, model_path=model_path, timesteps=timesteps, save=save, period=period, save_thresh=None)

    # TRAINING
    if not args.testing:
        print("#section Training")
        model = load_model_fn(partners=PARTNERS, testing=False)
        learn_model_fn(model, timesteps=args.timesteps, save=True, period=200)

    ts, period = 240, HP['n_steps_testing']

    # TESTING
    if args.testing:
        if args.baseline:   adapt_partner_baseline(load_model_fn, learn_model_fn, partners=PARTNERS, timesteps=ts, period=period, do_optimal=True)
        else:               adapt_partner_modular(load_model_fn, learn_model_fn, partners=PARTNERS, timesteps=ts, period=period, do_optimal=True)
        adapt_partner_scratch(load_model_fn, learn_model_fn, partners=PARTNERS, timesteps=ts, period=period, do_optimal=True)

if __name__ == "__main__":
    main()
