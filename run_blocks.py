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
from interactive_policy import BlocksPolicy
from partner_config import get_blocks_partners
from util import check_optimal, learn, load_model
from util import adapt_task, adapt_partner_baseline, adapt_partner_modular, adapt_partner_scratch

warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--n',              type=int, default=2,     help="Grid of size n x n")
parser.add_argument('--run',            type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--vis1',           type=int, default=1, choices=[1,2,3,4,5],     help="P1 visibility. 1) 100%, 2) 50%, 3) 0% 4) half chance of 100%, half chance of 0% 5) 50%")
parser.add_argument('--vis2',           type=int, default=3, choices=[1,2,3,4,5],     help="P2 visibility. 1) 100%, 2) 50%, 3) 0% 4) half chance of 100%, half chance of 0% 5) complement of P1")
parser.add_argument('--onesided',       action='store_true', default=False, help="True if we only consider reward of P2.")
parser.add_argument('--maxmovenumber',  type=int, default=6, choices=[2,4,6,8], help="Number of moves each game.")
parser.add_argument('--netsz',          type=int, default=80,     help="Size of policy network")
parser.add_argument('--latentz',        type=int, default=80,    help="Size of latent z dimension")

parser.add_argument('--mreg',           type=float, default=0.0, help="Marginal regularization.")
parser.add_argument('--baseline',       action='store_true', default=False, help="Baseline: no modular separation.")
parser.add_argument('--nomain',         action='store_true', default=False, help="Baseline: don't use main logits.")

parser.add_argument('--timesteps',      type=int, default=1000000,     help="Number of timesteps to train for")
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
        ('run={:04d}', run),
        ('vis1={:01d}', args.vis1),
        ('vis2={:01d}', args.vis2),
        ('onesided={:d}', args.onesided),
        ('mreg={:.2f}', mreg),
    ]

    m_name = '_'.join([t.format(v) for (t, v) in layout])
    m_path = 'output/blocks_' + m_name
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

if args.selfplay:
    PARTNERS = None     # this must be None to trigger selfplay
    HP['batch_size'] = 40
    HP['n_epochs'] = 10
else:
    setting, partner_type = "", "fixed" if args.fixedpartners else "ppo"
    TRAIN_PARTNERS, TEST_PARTNERS, INVERTTRAIN_PARTNERS, INVERTTEST_PARTNERS = get_blocks_partners(setting, partner_type)
    PARTNERS = [ TEST_PARTNERS[args.k % len(TEST_PARTNERS)] ] if args.testing else TRAIN_PARTNERS

def main():
    env = gym.make('blocks-v0', grid_size=args.n, vis1=args.vis1, vis2=args.vis2, one_sided_reward=args.onesided, max_move_number=args.maxmovenumber)
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
        return load_model(model_path=model_path, policy_class=BlocksPolicy, policy_kwargs=policy_kwargs, env=env, hp=HP, partners=partners, testing=testing, try_load=try_load)

    def learn_model_fn(model, timesteps, save, period):
        save_thresh = 19.2 if args.selfplay else None
        return learn(model, model_name=model_name, model_path=model_path, timesteps=timesteps, save=save, period=period, save_thresh=save_thresh)

    # TRAINING
    if not args.testing:
        print("#section Training")
        model = load_model_fn(partners=PARTNERS, testing=False)
        learn_model_fn(model, timesteps=args.timesteps, save=True, period=5000)

    ts, period = 25600, HP['n_steps_testing']
    # TESTING
    if args.testing and not args.zeroshot:
        if args.baseline:   adapt_partner_baseline(load_model_fn, learn_model_fn, partners=PARTNERS, timesteps=ts, period=period, do_optimal=False)
        else:               adapt_partner_modular(load_model_fn, learn_model_fn, partners=PARTNERS, timesteps=ts, period=period, do_optimal=False)

    if args.testing and args.zeroshot:
        adapt_task(load_model_fn, learn_model_fn, train_partners=TRAIN_PARTNERS, test_partners=TEST_PARTNERS, invert_train_partners=INVERTTRAIN_PARTNERS, invert_test_partners=INVERTTEST_PARTNERS, timesteps1=100000, timesteps2=500000, period=5000)

if __name__ == "__main__":
    main()
