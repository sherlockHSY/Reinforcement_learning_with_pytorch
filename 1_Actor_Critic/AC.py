# -*-coding:utf-8-*-
# @Time  : 2022/3/1 15:39
# @Author: Siyang
# @File  : 1_Actor_Critic.py
"""
Actor-Critic (1_Actor_Critic), Reinforcement Learning.

torch实现Actor-Critic算法
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple


seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_default_dtype(torch.float)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.save_actions = []
        # hidden layer
        n_layer = 20
        self.layer_1 = nn.Linear(state_dim, n_layer)
        self.output = nn.Linear(n_layer, action_dim)
        nn.init.normal_(self.layer_1.weight, 0., 0.1)
        nn.init.constant_(self.layer_1.bias, 0.1)
        nn.init.normal_(self.output.weight, 0., 0.1)
        nn.init.constant_(self.output.bias, 0.1)

    def forward(self, s):
        a1 = F.relu(self.layer_1(s))
        a = F.softmax(self.output(a1), dim=-1)
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        n_layer = 20
        self.layer_1 = nn.Linear(state_dim, n_layer)
        self.output = nn.Linear(n_layer, 1)
        nn.init.normal_(self.layer_1.weight, 0., 0.1)
        nn.init.constant_(self.layer_1.bias, 0.1)
        nn.init.normal_(self.output.weight, 0., 0.1)
        nn.init.constant_(self.output.bias, 0.1)

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        v = self.output(s)
        return v


class AC(object):
    def __init__(self, state_dim, action_dim, gamma=0.9, lr_a=0.001, lr_c=0.01):
        super(AC, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c

        # 定义 Actor 网络
        self.actor = Actor(state_dim, action_dim)

        # 定义 Critic 网络
        self.critic = Critic(state_dim, action_dim)

        # 定义优化器
        self.aopt = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.copt = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        # 选取损失函数
        self.mse_loss = nn.MSELoss()
        self.save_log_prob = []

    def choose_action(self, s):
        s = torch.FloatTensor(s)
        probs = self.actor(s)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        self.save_log_prob.append(log_prob)
        return action.item()

    def learn(self, s, a, r, s_):
        s = torch.FloatTensor(s)
        r = torch.FloatTensor([r])
        s_ = torch.FloatTensor(s_)
        # 训练critic
        v_eval = self.critic(s)
        v_ = self.critic(s_)
        v_target = r + self.gamma * v_
        td_error = v_target - v_eval
        loss = self.mse_loss(v_target, v_eval)  # TD_error = (r+gamma*V_next) - V_eval

        self.copt.zero_grad()
        loss.backward(retain_graph=True)

        # 训练Actor
        log_prob = self.save_log_prob.pop(-1)
        a_loss = -torch.mean(log_prob * td_error)
        self.aopt.zero_grad()
        a_loss.backward()
        self.copt.step()
        self.aopt.step()

