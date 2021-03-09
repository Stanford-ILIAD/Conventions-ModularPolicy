import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

def check_optimal(env, model, n_eval_episodes, trunc=None):
    model.policy.setup_optimal_mask(env)
    wass_dist_main, wass_dist_partner, cnt = 0, 0, 0

    for partner_idx in range(model.policy.num_partners):
        for _ in range(n_eval_episodes):
            obs = env.reset()
            done, state = False, None
            while not done:
                thobs = th.Tensor([obs])
                # compare wasserstein distance between main_logits and optimal mask
                main_logits, partner_logits = model.policy.get_action_logits_from_obs(obs=thobs, partner_idx=partner_idx)
                if trunc is not None: main_logits, partner_logits = main_logits[:, :trunc], partner_logits[:, :trunc]
                main_logits, partner_logits = main_logits - main_logits.logsumexp(dim=-1, keepdim=True), partner_logits - partner_logits.logsumexp(dim=-1, keepdim=True)
                main_prob, partner_prob = th.exp(main_logits), th.exp(partner_logits)

                optimal_mask = model.policy.get_mask(thobs).float()
                if not optimal_mask.bool().all():
                    if trunc is not None: optimal_mask = optimal_mask[:, :trunc]
                    optimal_prob = optimal_mask / optimal_mask.sum(dim=-1, keepdim=True)
                    #print(obs, main_prob, partner_prob, optimal_prob)

                    wass_dist_main += th.abs(main_prob - optimal_prob).sum().data
                    wass_dist_partner += th.abs(partner_prob - optimal_prob).sum().data
                    cnt += 1.0

                action, _ = model.predict(observation=obs, partner_idx=partner_idx, deterministic=False)
                obs, reward, done, _info = env.step(action)

    print(wass_dist_main / cnt, wass_dist_partner / cnt)


def learn(model, model_name, model_path, timesteps, save, period, save_thresh=None):
    # TRAINING
    steps, save_period, loop_period = timesteps, 10*period, period

    while True:
        for partner_idx in range(model.policy.num_partners):
            mean_reward, std_reward = evaluate_policy(model, model.get_env(), partner_idx=partner_idx, n_eval_episodes=200, deterministic=False)
            print("#data steps %u\t partner %u\t mean_rew %.2f\t std_rew %.2f\t" % (timesteps - steps, partner_idx, mean_reward, std_reward))

        if save_thresh is not None and mean_reward >= save_thresh:
            model.save(model_path), print("saving...")
            break

        if steps == 0: break
        cur_steps = min(steps, loop_period)
        steps -= cur_steps

        model.learn(total_timesteps=cur_steps, tb_log_name=model_name, reset_num_timesteps=False)
        time_to_save = steps == 0 or steps // save_period != (steps+cur_steps) // save_period
        if save and time_to_save: model.save(model_path), print("saving...")

def load_model(model_path, policy_class, policy_kwargs, env, hp, partners, testing, try_load=True):
    load_successful = False

    if try_load:
        try:
            model = PPO.load(model_path)#, policy_kwargs=policy_kwargs)
            load_successful = True
            print("Model loaded successfully")
        except Exception as e:
            print("Could not load model", e)

    if not load_successful:
        print("Create new model")
        
        n_steps, batch_size, n_epochs, = hp['n_steps'], hp['batch_size'], hp['n_epochs']
        model = PPO(policy_class, env, policy_kwargs=policy_kwargs, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, verbose=0, ent_coef=0.00, marginal_reg_coef=hp['mreg'])

        for name, param in model.policy.named_parameters():
            if param.requires_grad:
                print(name, param.data.size())

    vec_env = DummyVecEnv([lambda: env])
    model.set_env(vec_env)

    model.policy.set_partners(partners)
    if testing:            
        model.policy.num_partners       = 1 # only test 1 partner
        model.marginal_reg_coef    = 0
        model.n_epochs                  = hp['n_epochs_testing']
        model.n_steps                   = hp['n_steps_testing']
        model._init_rollout_buffer()
        
    return model


def adapt_task(load_model_fn, learn_model_fn, train_partners, test_partners, invert_train_partners, invert_test_partners, timesteps1, timesteps2, period):
    if len(test_partners) > len(train_partners):    test_partners = test_partners[:len(train_partners)]

    # ADAPT TASK
    # new partners
    model = load_model_fn(test_partners, testing=False)
    model.policy.do_init_weights(init_main=False, init_partner=True)
    model.policy.set_freeze_main(freeze=True)
    model.policy.set_freeze_partner(freeze=False)
    learn_model_fn(model, timesteps1, save=False, period=period)

    # invert task on new partners
    model.policy.set_partners(invert_test_partners)
    model.env.envs[0].set_invert(invert=True)
    model.policy.do_init_weights(init_main=False, init_partner=False)
    model.policy.set_freeze_main(freeze=False)
    model.policy.set_freeze_partner(freeze=True)
    learn_model_fn(model, timesteps2, save=False, period=period)

    # use new main module on orig partners
    print("#section AdaptTask")
    model_orig = load_model_fn(invert_train_partners, testing=False)
    model_orig.policy.overwrite_main(model.policy)
    model_orig.env.envs[0].set_invert(invert=True)         # invert task on orig partners
    learn_model_fn(model_orig, 1, save=False, period=1)


def adapt_partner_baseline(load_model_fn, learn_model_fn, partners, timesteps, period, do_optimal=False):
    model = load_model_fn(partners, testing=True)
    print("#section AdaptPartnerBaseline")
    model.policy.set_freeze_main(freeze=False)
    model.policy.set_freeze_partner(freeze=False)
    learn_model_fn(model, timesteps, save=False, period=period)

    if do_optimal:
        model = load_model_fn(partners, testing=True)
        model.policy.setup_optimal_mask(model.env.envs[0])
        print("#section AdaptPartnerBaselineOptimal")
        model.policy.set_freeze_main(freeze=False)
        model.policy.set_freeze_partner(freeze=False)
        learn_model_fn(model, timesteps, save=False, period=period)

def adapt_partner_modular(load_model_fn, learn_model_fn, partners, timesteps, period, do_optimal=False):
    model = load_model_fn(partners, testing=True)
    print("#section AdaptPartner")
    model.policy.do_init_weights(init_main=False, init_partner=True)
    model.policy.set_freeze_main(freeze=False)
    model.policy.set_freeze_partner(freeze=False)
    learn_model_fn(model, timesteps, save=False, period=period)

    if do_optimal:
        model = load_model_fn(partners, testing=True)
        model.policy.setup_optimal_mask(model.env.envs[0])
        print("#section AdaptPartnerOptimal")
        model.policy.do_init_weights(init_main=False, init_partner=True)
        model.policy.set_freeze_main(freeze=False)
        model.policy.set_freeze_partner(freeze=False)
        learn_model_fn(model, timesteps, save=False, period=period)

def adapt_partner_scratch(load_model_fn, learn_model_fn, partners, timesteps, period, do_optimal=False):
    model = load_model_fn(partners, testing=True, try_load=False)
    print("#section AdaptPartnerFromScratch")
    model.policy.do_init_weights(init_main=True, init_partner=True)
    model.policy.set_freeze_main(freeze=False)
    model.policy.set_freeze_partner(freeze=False)
    learn_model_fn(model, timesteps, save=False, period=period)

    if do_optimal:
        model = load_model_fn(partners, testing=True, try_load=False)
        model.policy.setup_optimal_mask(env=model.env.envs[0])
        print("#section AdaptPartnerFromScratchOptimal")
        model.policy.do_init_weights(init_main=True, init_partner=True)
        model.policy.set_freeze_main(freeze=False)
        model.policy.set_freeze_partner(freeze=False)
        learn_model_fn(model, timesteps, save=False, period=period)
