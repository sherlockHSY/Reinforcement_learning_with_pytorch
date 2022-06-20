# -*-coding:utf-8-*-
# @Time  : 2022/3/1 15:39
# @Author: hsy
# @File  : Actor_Critic.py
"""
Actor-Critic (Actor_Critic), Reinforcement Learning.

torch实现 Advantage Actor-Critic算法即A2C
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils.simplebuffer import SimpleBuffer


class Actor(nn.Module):
    """ 2层MLP """
    def __init__(self, state_dim, action_dim, hidden_dim_1, hidden_dim_2):
        super(Actor, self).__init__()
        self.save_actions = []
        self.layer_1 = nn.Linear(state_dim, hidden_dim_1)
        self.layer_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.output = nn.Linear(hidden_dim_2, action_dim)

    def forward(self, s):
        a1 = F.relu(self.layer_1(s))
        a2 = F.relu(self.layer_2(a1))
        a = F.softmax(self.output(a2), dim=1)
        return a


class Critic(nn.Module):
    """ 2层MLP,输出状态价值 """
    def __init__(self, state_dim, action_dim, hidden_dim_1, hidden_dim_2):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim, hidden_dim_1)
        self.layer_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.output = nn.Linear(hidden_dim_2, 1)

    def forward(self, s):
        s1 = F.relu(self.layer_1(s))
        s2 = F.relu(self.layer_2(s1))
        v = self.output(s2)
        return v


class AdvantageActorCritic(object):
    def __init__(self, state_dim, action_dim, config, device):
        super(AdvantageActorCritic, self).__init__()
        self.name = "Advantage_Actor_Critic"
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim_1 = config["hidden_dim_1"]
        self.hidden_dim_2 = config["hidden_dim_2"]
        self.gamma = config["gamma"]
        self.lr_a = config["lr_actor"]
        self.lr_c = config["lr_critic"]
        self.buffer_size = config["buffer_size"]
        # 定义 Actor 网络
        self.actor = Actor(state_dim, action_dim, self.hidden_dim_1, self.hidden_dim_2).to(device)

        # 定义 Critic 网络
        self.critic = Critic(state_dim, action_dim, self.hidden_dim_1, self.hidden_dim_2).to(device)

        # 定义优化器
        self.aopt = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.copt = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
        # 选取损失函数
        self.mse_loss = nn.MSELoss()

        self.buffer = SimpleBuffer(self.buffer_size, device)

    def select_action(self, s):
        s = torch.tensor(np.array([s]), dtype=torch.float).to(self.device)
        probs = self.actor(s)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item()

    def update(self):
        # sample data
        states, actions, rewards, next_states, dones = self.buffer.sample()

        # 训练actor
        v_eval = self.critic(states)
        v_ = self.critic(next_states)
        v_target = rewards + self.gamma * v_ * (1-dones)
        td_error = v_target - v_eval
        log_probs = torch.log(self.actor(states).gather(1, actions))
        # 这里可以actor不是直接拿Q值来作为loss，实际是变成拟合状态值函数来拟合优势函数即变成 V(s')-V(s)
        actor_loss = torch.mean(-log_probs * td_error.detach())

        # 训练critic
        critic_loss = torch.mean(self.mse_loss(v_eval, v_target.detach()))
        self.aopt.zero_grad()
        self.copt.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.aopt.step()
        self.copt.step()
        self.buffer.clear()
        return actor_loss.clone().data.numpy() if self.device == torch.device('cpu') else actor_loss.clone().data.cpu().numpy()