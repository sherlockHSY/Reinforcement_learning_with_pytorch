# -*-coding:utf-8-*-
# @Time  : 2022/3/2 11:04
# @Author: hsy
# @File  : ddqn.py
"""
Double Deep Q Learning (DDQN), Reinforcement Learning.
DDQN算法，结构与DQN一样，唯一的不同在于修改了Q值更新的过程。
support gqu.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.replaybuffer import ReplayBuffer


class DQNetwork(nn.Module):
	""" 2层MLP """
	def __init__(self, state_dim, action_dim, hidden_dim_1, hidden_dim_2):
		super(DQNetwork, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim

		# full-connected layer
		self.fc1 = nn.Linear(self.state_dim, hidden_dim_1)
		self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
		self.output = nn.Linear(hidden_dim_2, action_dim)

	def forward(self, x):
		# use ReLU as activate function
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))

		q_value = self.output(x)

		return q_value


# Double DQN
class DoubleDQN(object):

	def __init__(self, state_dim, action_dim, config, device):
		super(DoubleDQN, self).__init__()
		self.name = "DoubleDQN"
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.lr = config['learning_rate']
		self.epsilon = config['epsilon']
		self.epsilon_decay = config['epsilon_decay']
		self.min_epsilon = config['min_epsilon']
		self.gamma = config['gamma']
		self.tau = config['tau']
		self.batch_size = config['batch_size']
		self.buffer_size = config['buffer_size']
		self.hidden_dim_1 = config['hidden_dim_1']
		self.hidden_dim_2 = config['hidden_dim_2']
		self.device = device  # set device to cpu or gpu

		# Network
		self.eval_net = DQNetwork(state_dim, action_dim, self.hidden_dim_1, self.hidden_dim_2).to(device)
		# Target network
		self.target_net = DQNetwork(state_dim, action_dim, self.hidden_dim_1, self.hidden_dim_2).to(device)
		self.target_net.load_state_dict(self.eval_net.state_dict())
		self.optimizer = torch.optim.Adam(self.eval_net.parameters(), self.lr)  # Adam Optimizer
		self.loss_func = nn.MSELoss()
		# Experience Memory Replay
		self.buffer = ReplayBuffer(self.buffer_size, self.batch_size, device)

	def select_action(self, s):
		# epsilon greedy
		if np.random.random() < self.epsilon:
			action = np.random.randint(self.action_dim)
		else:
			s = torch.tensor(np.array([s]), dtype=torch.float).to(self.device)
			action_prob = self.eval_net(s)
			action = torch.argmax(action_prob).item()

		return action

	def decrement_epsilon(self):
		"""
		衰减贪心程度，前期多些探索，后期减小探索
		Decrements the epsilon after each step till it reaches minimum epsilon
		epsilon = epsilon - decrement (default is 0.99e-6)
		"""
		self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.min_epsilon \
			else self.min_epsilon

	def update(self):
		# Sample data
		states, actions, rewards, states_, terminals = self.buffer.sample()
		actions = actions.squeeze(dim=-1)
		rewards = rewards.squeeze(dim=-1)
		terminals = terminals.squeeze(dim=-1)

		batches = np.arange(self.batch_size)

		q_eval = self.eval_net(states)[batches, actions]

		q_next = self.target_net.forward(states_)

		# -----------------Here is the only difference from DQN--------------------
		q_pred = self.eval_net.forward(states_)
		max_action = torch.argmax(q_pred, dim=1)
		max_q_next = q_next[batches, max_action]
		max_q_next[terminals] = 0.0
		# ---------------------------------------------------------------------
		q_target = rewards + self.gamma * max_q_next

		loss = self.loss_func(q_target, q_eval).to(self.device)
		loss_np = loss.clone().data.numpy() if self.device == torch.device('cpu') else loss.clone().data.cpu().numpy()

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		self.decrement_epsilon()  # decay epsilon

		# soft update target net 学习一段时间更新目标网络
		# soft的意思是每次learn的时候更新部分参数
		for target_param, eval_param in zip(self.target_net.parameters(), self.eval_net.parameters()):
			target_param.data.copy_(self.tau * eval_param.data + (1.0 - self.tau) * target_param.data)

		return loss_np


