# -*- coding:utf-8  -*-
# Time  : 2021/5/31 下午4:14
# Author: Yahui Cui

"""
# =================================== Important =========================================
Notes:
1. this agent is random agent , which can fit any env in Jidi platform.
2. if you want to load .pth file, please follow the instruction here:
https://github.com/jidiai/ai_lib/blob/master/examples/demo
"""

import argparse
import os
import sys
from agents.ppo.ppo import PPO
from agents.ppo.singleagent import SingleRLAgent


agent = PPO()
agent.load(os.path.dirname(os.path.abspath(__file__)), 500)

sys.path.pop(-1)  # just for safety

def action_from_algo_to_env(action_space, is_act_continuous, joint_action):
    """
    :param joint_action:
    :return: wrapped joint action: one-hot
    """
    res = []

    if is_act_continuous:
        res = [[joint_action[0]], [10]]
    else:
        # print(action_space)
        nvec1 = action_space[0].high - action_space[0].low + 1
        nvec2 = action_space[1].high - action_space[1].low + 1
        # print(nvec1, nvec2)
        
        each1 = [0] * int(nvec1[0])
        each2 = [0] * int(nvec2[0])

        each1[joint_action[0]] = 1
        # each2[joint_action[1]] = 1
        res = [each1, [10]]
    
    # print(res)
    return res

def my_controller(observation, action_space, is_act_continuous=False):
    observation_copy = observation.copy()
    observation = observation_copy["obs"]['agent_obs']
    agent_id = observation_copy["controlled_player_index"]
    # print('agent', agent_id)
    observation = observation.flatten()
    # print(observation)
    action_from_algo = agent.select_action(observation, False)
    # print('action', action_from_algo)
    action_to_env = action_from_algo_to_env(action_space, is_act_continuous, action_from_algo)
    return action_to_env

# def my_controller(observation, action_space, is_act_continuous=False):
#     # agent_action = []
#     # for i in range(len(action_space)):
#     print(observation['obs'])
#     obs_ctrl_agent, energy_ctrl_agent = observation['obs']['agent_obs'].flatten(), observation['obs']['energy']
#     # obs_oppo_agent, energy_oppo_agent = state[1-ctrl_agent_index]['agent_obs'], env.agent_list[1-ctrl_agent_index].energy
#     agent_action = agent.select_action(obs_ctrl_agent, False)
#         # agent_action.append(action_)
#     print(agent_action)
#     return agent_action

# def my_controller(observation, action_space, is_act_continuous=False):
#     agent_action = agent.select_action(observation, False)
#     return agent_action


def sample_single_dim(action_space_list_each, is_act_continuous):
    each = []
    if is_act_continuous:
        each = action_space_list_each.sample()
    else:
        if action_space_list_each.__class__.__name__ == "Discrete":
            each = [0] * action_space_list_each.n
            idx = action_space_list_each.sample()
            each[idx] = 1
        elif action_space_list_each.__class__.__name__ == "MultiDiscreteParticle":
            each = []
            nvec = action_space_list_each.high - action_space_list_each.low + 1
            sample_indexes = action_space_list_each.sample()

            for i in range(len(nvec)):
                dim = nvec[i]
                new_action = [0] * dim
                index = sample_indexes[i]
                new_action[index] = 1
                each.extend(new_action)
    return each


def sample(action_space_list_each, is_act_continuous):
    player = []
    if is_act_continuous:
        for j in range(len(action_space_list_each)):
            each = action_space_list_each[j].sample()
            player.append(each)
    else:
        player = []
        for j in range(len(action_space_list_each)):
            # each = [0] * action_space_list_each[j]
            # idx = np.random.randint(action_space_list_each[j])
            if action_space_list_each[j].__class__.__name__ == "Discrete":
                each = [0] * action_space_list_each[j].n
                idx = action_space_list_each[j].sample()
                each[idx] = 1
                player.append(each)
            elif action_space_list_each[j].__class__.__name__ == "MultiDiscreteParticle":
                each = []
                nvec = action_space_list_each[j].high
                sample_indexes = action_space_list_each[j].sample()

                for i in range(len(nvec)):
                    dim = nvec[i] + 1
                    new_action = [0] * dim
                    index = sample_indexes[i]
                    new_action[index] = 1
                    each.extend(new_action)
                player.append(each)
    return player