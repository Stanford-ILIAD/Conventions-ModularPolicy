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
from partner_config import get_arms_partners
from util import check_optimal, learn, load_model
from util import adapt_task, adapt_partner_baseline, adapt_partner_modular, adapt_partner_scratch

warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--n',              type=int, default=4,     help="n contexts, 2n arms")
parser.add_argument('--m',              type=int, default=1,     help="m contexts with rules")
parser.add_argument('--run',            type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--netsz',          type=int, default=30,    help="Size of policy network")
parser.add_argument('--latentz',        type=int, default=30,    help="Size of latent z dimension")

parser.add_argument('--mreg',           type=float, default=0.0, help="Marginal regularization.")
parser.add_argument('--baseline',       action='store_true', default=False, help="Baseline: no modular separation.")
parser.add_argument('--nomain',         action='store_true', default=False, help="Baseline: don't use main logits.")

parser.add_argument('--timesteps',      type=int, default=10000,     help="Number of timesteps to train for")
parser.add_argument('--ppopartners',    action='store_true', default=False, help="use ppo partners")
parser.add_argument('--fixedpartners',  action='store_true', default=False, help="use fixed partners")
parser.add_argument('--selfplay',       action='store_true', default=False, help="converge using selfplay")
parser.add_argument('--testing',        action='store_true', default=False, help="Testing.")
parser.add_argument('--zeroshot',       action='store_true', default=False, help="Try zeroshot combination of task + partner.")

parser.add_argument('--k',              type=int, default=0, help="When fixedpartner=True, k is the index of the test partner")

args = parser.parse_args()
print(args)
assert((bool)(args.ppopartners) + (bool)(args.fixedpartners) + (bool)(args.selfplay) == 1)

def get_model_name_and_path(run, mreg=0.00):
    layout = [
        ('n={:01d}', args.n),
        ('m={:01d}', args.m),
        ('run={:04d}', run),
        ('netsz={:03d}',  args.netsz),
        ('mreg={:.2f}', mreg),
    ]

    m_name = '_'.join([t.format(v) for (t, v) in layout])
    m_path = 'output/arms_' + m_name
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

if args.selfplay:
    PARTNERS = None     # this must be None to trigger selfplay
else:
    setting, partner_type = "n%um%u" % (args.n, args.m), "fixed" if args.fixedpartners else "ppo"
    TRAIN_PARTNERS, TEST_PARTNERS, INVERTTRAIN_PARTNERS, INVERTTEST_PARTNERS = get_arms_partners(setting, partner_type)
    PARTNERS = [ TEST_PARTNERS[args.k % len(TEST_PARTNERS)] ] if args.testing else TRAIN_PARTNERS

def main():
    global PARTNERS
    env = gym.make('arms-v0', n=args.n, m=args.m)
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
        save_thresh = 0.95 if args.selfplay else None
        return learn(model, model_name=model_name, model_path=model_path, timesteps=timesteps, save=save, period=period, save_thresh=save_thresh)

    # TRAINING
    if not args.testing:
        print("#section Training")
        model = load_model_fn(partners=PARTNERS, testing=False)
        learn_model_fn(model, timesteps=args.timesteps, save=True, period=200)

    ts, period = 240, HP['n_steps_testing']

    # TESTING
    if args.testing and not args.zeroshot:
        if args.baseline:   adapt_partner_baseline(load_model_fn, learn_model_fn, partners=PARTNERS, timesteps=ts, period=period, do_optimal=True)
        else:               adapt_partner_modular(load_model_fn, learn_model_fn, partners=PARTNERS, timesteps=ts, period=period, do_optimal=True)
        adapt_partner_scratch(load_model_fn, learn_model_fn, partners=PARTNERS, timesteps=ts, period=period, do_optimal=True)

    if args.testing and args.zeroshot:
        adapt_task(load_model_fn, learn_model_fn, train_partners=TRAIN_PARTNERS, test_partners=TEST_PARTNERS, invert_train_partners=INVERTTRAIN_PARTNERS, invert_test_partners=INVERTTEST_PARTNERS, timesteps1=2000, timesteps2=6000, period=1000)

if __name__ == "__main__":
    main()
