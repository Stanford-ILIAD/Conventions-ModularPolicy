import numpy as np
import torch as th

def tabular_q_learning(env, discount_rate=0.01, step_size=0.01, eps=1e-4):
    action_space = env.action_space.nvec # multi-discrete
    state_space = env.observation_space.nvec # multi-discrete
    q_values = np.zeros( np.concatenate((state_space, action_space)) , np.float64)

    for _ in range(1000):
        for state_action, q_val in np.ndenumerate(q_values):
            state, action = state_action[:len(state_space)], state_action[len(state_space):]
            #print("state, action: ", state, action)
            
            env.reset(list(state))
            next_state, reward, done, _ = env.step(action)

            delta = reward - q_values[state_action]
            if not done: 
                next_state = tuple(next_state)
                delta = reward + discount_rate * q_values[next_state].max() - q_values[state_action]
            q_values[state_action] += step_size * delta

    q_values = th.tensor(q_values)

    maxout_actions = th.max(th.max(q_values, dim=-2, keepdim=True).values, dim=-1, keepdim=True).values
    maxout_action2 = th.max(q_values, dim=-1, keepdim=True).values
    maxout_action1 = th.max(q_values, dim=-2, keepdim=True).values

    optimal_action1_mask = (maxout_action2 - maxout_actions > -eps).squeeze(-1)
    optimal_action2_mask = (maxout_action1 - maxout_actions > -eps).squeeze(-2)

    # print(q_values)
    print(optimal_action1_mask)
    print(optimal_action2_mask)
    return q_values, optimal_action1_mask, optimal_action2_mask
