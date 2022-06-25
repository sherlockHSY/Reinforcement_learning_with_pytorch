# -*-coding:utf-8-*-
# @Time  : 2022/4/23 17:05
# @Author: hsy
# @File  : train_cartpole.py
"""
Cartpole is an environment with discrete actions,
We use this environment to test rl algorithms which support discrete action.

Model-free: A2C, DQN, DDQN, DuelingDQN, D3QN, PPO, SAC
Model-based: MBPO
"""
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import gym

from model_free.Actor_Critic.a2c import AdvantageActorCritic as A2C

from model_free.DQN.dqn import DQN
from model_free.Double_DQN.ddqn import DoubleDQN as DDQN
from model_free.Dueling_DQN.dueling_dqn import DuelingDQN
from model_free.Dueling_Double_DQN.d3qn import D3QN
# from Dueling_Double_DQN.d3qn_prioritized_replay import D3QN as D3QN_PER

from model_free.Soft_Q_Learning.SoftQLearning import SoftQLearning
from model_free.SAC.sac_discrete import SAC_Discrete
from model_free.SAC.sac_discrete_2 import SAC_Discrete as SAC_Discrete_2

from model_free.PPO.ppo import PPO

from model_based.MBPO.mbpo_discrete import MBPO_Discrete


parser = argparse.ArgumentParser(description="data")
parser.add_argument('-c', '--cuda', type=int, default=0)  # 如果有多张卡，可指定卡号
parser.add_argument('-a', '--agent', type=str, default='PPO')
parser.add_argument('-me', '--max_episodes', type=int, default=2000)
args = parser.parse_args()

cuda = args.cuda
agent_name = args.agent

# set device to gpu if could
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:' + str(cuda))
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

# Training Hyper Parameters
max_episodes = args.max_episodes
max_ep_len = 1000
max_scores = 1000
seed = 1

# Algorithm Hyper Parameters
hyperparameters = {
    "DQN_Agents": {
        "gamma": 0.99,  # discount factor
        "epsilon": 1.0,  # epsilon greedy
        "min_epsilon": 0.001,
        "epsilon_decay": 0.001,
        "learning_rate": 0.0005,
        "tau": 0.05,  # soft update target network
        "hidden_dim_1": 64,  # first hidden layer dimension
        "hidden_dim_2": 32,  # second hidden layer dimension
        "batch_size": 64,
        "buffer_size": 100000,
        "update_mode": "single-step",  # update mode: episode(update by one episode),
                                       # single-step(update by one single step),
                                       # multi-step(update by multi steps)
        "begin_update_timestep": 500,
        "update_timestep": 100,  # update policy every n timesteps
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.1,
        "incremental_td_error": 1e-8,
        "batch_norm": False,
        "gradient_clipping_norm": 0.7,
        "learning_iterations": 1,
        "clip_rewards": False,
    },
    "SAC": {
        "gamma": 0.99,  # discount factor
        "tau": 0.005,  # soft update target network
        "lr_alpha": 0.01,  # learning rate for temperature Coefficient alpha
        "lr_actor": 0.001,  # learning rate for actor network
        "lr_critic": 0.01,  # learning rate for critic network
        "hidden_dim_1": 128,  # first hidden layer dimension
        "hidden_dim_2": 32,  # second hidden layer dimension
        "batch_size": 64,
        "buffer_size": 100000,
        "update_mode": "single-step",  # update mode: episode(update by one episode),
                                       # single-step(update by one single step),
                                       # multi-step(update by multi steps)
        "begin_update_timestep": 500,  #
        "update_timestep": 50,  # update policy every n timesteps
        "clip_grad_param": 1,
        "target_entropy": -1,  # 目标熵，H=-dim|A|
    },
    "Policy_Gradient_Agents": {
        "gamma": 0.99,  # discount factor
        "lr_actor": 0.001,  # learning rate for actor network
        "lr_critic": 0.01,  # learning rate for critic network
        "learning_rate": 0.05,
        "hidden_dim_1": 64,  # first hidden layer dimension
        "hidden_dim_2": 32,  # second hidden layer dimension
        "has_continuous_action_space": False,
        "action_std": 0.6,                    # starting std for action distribution (Multivariate Normal)
        "action_std_init": 0.6,
        "min_action_std": 0.1,  # minimum action_std (stop decay after action_std <= min_action_std)
        "action_std_decay_rate": 0.05,  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        "action_std_decay_freq": int(2.5e5),  # action_std decay frequency (in num timesteps)
        "begin_update_timestep": 500,  # begin policy update after n timesteps
        "update_timestep": 100,      # update policy every n timesteps
        "K_epochs": 80,               # update policy for K epochs in one PPO update
        "eps_clip": 0.2,              # clip parameter for PPO
        "update_mode": "multi-step",  # update mode: episode(update by one episode),
                                      # single-step(update by one single step),
                                      # multi-step(update by multi steps)
        "normalise_rewards": False,
        "gradient_clipping_norm": 7.0,
        "mu": 0.0,     # only required for continuous action games
        "theta": 0.0,  # only required for continuous action games
        "sigma": 0.0,  # only required for continuous action games
        "epsilon_decay_rate_denominator": 1.0,
        "clip_rewards": False
    },

    "Actor_Critic_Agents":  {
        "gamma": 0.99,  # discount factor
        "tau": 0.005,  # soft update target network
        "learning_rate": 0.005,  # learning rate
        "lr_actor": 0.001,  # learning rate for actor network
        "lr_critic": 0.01,  # learning rate for critic network
        "hidden_dim_1": 64,  # first hidden layer dimension
        "hidden_dim_2": 32,  # second hidden layer dimension
        "buffer_size": 100000,
        "update_mode": "episode",  # update mode: episode(update by one episode),
                                   # single-step(update by one single step),
                                   # multi-step(update by multi steps)
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": 5.0,
        "epsilon_decay_rate_denominator": 1.0,
        "normalise_rewards": True,
        "exploration_worker_difference": 2.0,
        "clip_rewards": False,

        "Actor": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": "Softmax",
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },
        "Critic": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "min_steps_before_learning": 400,
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0, #for O-H noise
        "theta": 0.15, #for O-H noise
        "sigma": 0.25, #for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True
    },
    "Stochastic_Policy_Search_Agents": {
        "policy_network_type": "Linear",
        "noise_scale_start": 1e-2,
        "noise_scale_min": 1e-3,
        "noise_scale_max": 2.0,
        "noise_scale_growth_factor": 2.0,
        "stochastic_action_decision": False,
        "num_policies": 10,
        "episodes_per_policy": 1,
        "num_policies_to_keep": 5,
        "clip_rewards": False
    },
    "MBPO": {
        "real_ratio": 0.5,
        "model_alpha": 0.01,  # 模型损失函数中的加权权重
        "rollout_batch_size": 1000,
        "rollout_length": 1,  # 推演长度k,推荐更多尝试
        "model_pool_size": 1000,  # rollout_batch_size * rollout_length

        "gamma": 0.99,  # discount factor
        "tau": 0.005,  # soft update target network
        "lr_alpha": 0.01,  # learning rate for temperature Coefficient alpha
        "lr_actor": 0.001,  # learning rate for actor network
        "lr_critic": 0.01,  # learning rate for critic network
        "hidden_dim_1": 64,  # first hidden layer dimension
        "hidden_dim_2": 32,  # second hidden layer dimension
        "batch_size": 64,
        "buffer_size": 100000,
        "update_mode": "single-step",  # update mode: episode(update by one episode),
                                       # single-step(update by one single step),
                                       # multi-step(update by multi steps)
        "begin_update_timestep": 500,  #
        "update_timestep": 50,  # update policy every n timesteps
        "target_entropy": -1,  # 目标熵，H=-dim|A|

    }
}
# agent name -> group,class
name_to_group_class_dictionary = {
        "DQN": {"group": "DQN_Agents", "class": DQN},
        "DDQN": {"group": "DQN_Agents", "class": DDQN},
        "DDQN_PER": {"group": "DQN_Agents", "class": DDQN},
        "DuelingDQN": {"group": "DQN_Agents", "class": DuelingDQN},
        "D3QN": {"group": "DQN_Agents", "class": D3QN},
        "D3QN_PER": {"group": "DQN_Agents", "class": D3QN},
        "REINFORCE": {"group": "Policy_Gradient_Agents", "class": D3QN},
        "PPO": {"group": "Policy_Gradient_Agents", "class": PPO},
        "DDPG": {"group": "Actor_Critic_Agents", "class": D3QN},
        "A2C": {"group": "Actor_Critic_Agents", "class": A2C},
        "A3C": {"group": "Actor_Critic_Agents", "class": D3QN},
        "SQL": {"group": "SAC", "class": SoftQLearning},
        "SAC": {"group": "SAC", "class": SAC_Discrete},
        "SAC_Discrete": {"group": "SAC", "class": SAC_Discrete},
        "SAC_Discrete_2": {"group": "SAC", "class": SAC_Discrete_2},
        "MBPO_Discrete": {"group": "MBPO", "class": MBPO_Discrete}
    }
# init environment
env_name = "CartPole-v1"
env = gym.make(env_name).unwrapped
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print("environment: ", env_name)
print("state space dimension : ", state_dim)
print("action space dimension : ", action_dim)
# set random seed
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# init agent
agent_group = name_to_group_class_dictionary[agent_name]["group"]
agent_class = name_to_group_class_dictionary[agent_name]["class"]
agent_config = hyperparameters[agent_group]
agent = agent_class(state_dim=state_dim,
                    action_dim=action_dim,
                    config=agent_config,
                    device=device)

update_mode = agent_config["update_mode"]
# Writer will output to ./runs/ directory by default
dir_name = "./runs/cartpole/" + agent_name + '_' + update_mode + '_update/'
writer = SummaryWriter(dir_name)

# begin training
print('==============================================================================')
print('training with {} '.format(agent_name))
i_episode, i_update, time_step = 0, 0, 0
while i_episode < max_episodes:
    state = env.reset()
    ep_reward = 0
    track_r = []
    t = 0
    while True:
        # select action
        action = agent.select_action(state)

        state_, reward, done, info = env.step(action)

        agent.buffer.add(state, action, reward, state_, done)
        ep_reward += reward
        track_r.append(reward)
        state = state_
        time_step += 1
        t += 1

        if agent_config["update_mode"] == "episode":
            if done:
                loss = agent.update()
                writer.add_scalar('Loss', loss, i_update)
                i_update += 1
        elif agent_config["update_mode"] == "single-step":
            if time_step > agent_config["begin_update_timestep"]:
                loss = agent.update()
                writer.add_scalar('Loss', loss, i_update)
                i_update += 1
        elif agent_config["update_mode"] == "multi-step":
            if time_step % agent_config["update_timestep"] == 0:
                loss = agent.update()
                writer.add_scalar('Loss', loss, i_update)
                i_update += 1

        if done or t >= max_ep_len:
            break

        # print some training info
        if time_step % 1000 == 0:
            print('--{} already iteract {} steps! and update {} times!------'.format(agent_name, time_step, i_update))


    writer.add_scalar('Reward', ep_reward, i_episode)

    print("Episode: ", i_episode, "  ep reward: ", round(ep_reward, 2))

    i_episode += 1

writer.close()
