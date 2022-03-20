# -*-coding:utf-8-*-
# @Time  : 2022/3/2 11:04
# @Author: hsy
# @File  : my_dqn.py
"""
Deep Q Learning (DQN), Reinforcement Learning.
只适合离散动作的DQN算法,可在gpu上跑
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# device
# set device to cpu or cuda
device = torch.device('cpu')
if torch.cuda.is_available():
	device = torch.device('cuda:0')
	torch.cuda.empty_cache()
	print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
	print("Device set to : cpu")


class DQN(object):
	def __init__(self, state_dim, action_dim, batch_size, lr, epsilon, gamma, memory_capacity, target_replace_iter):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.lr = lr
		self.epsilon = epsilon
		self.gamma = gamma
		self.memory_capacity = memory_capacity
		self.target_replace_iter = target_replace_iter
		self.batch_size = batch_size
		# Memory
		self.memory = np.zeros((memory_capacity, state_dim * 2 + 1 + 1))
		self.pointer = 0

		self.update_target_step = 0   # for update target net

		self.eval_net = nn.Sequential(
			nn.Linear(state_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, action_dim)
		).to(device)

		self.target_net = nn.Sequential(
			nn.Linear(state_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, action_dim)
		).to(device)

		self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)    # Adam Optimizer
		self.loss_func = nn.MSELoss()

	def choose_action(self, s):
		s = torch.FloatTensor(s).to(device)
		s = torch.unsqueeze(s, 0).to(device)

		# epsilon-greedy
		if np.random.uniform() < self.epsilon:
			action_prob = self.eval_net(s)
			if device == torch.device('cpu'):
				action = torch.max(action_prob, 1)[1].data.numpy()
			else:
				action = torch.max(action_prob, 1)[1].data.cpu().numpy()
			action = action[0]
		else:
			action = np.random.randint(0, self.action_dim)

		return action

	def store_transition(self, s, a, r, s_):
		transition = np.hstack((s, a, [r], s_))
		index = self.pointer % self.memory_capacity
		self.memory[index, :] = transition
		self.pointer += 1

	def sample(self):
		indices = np.random.choice(self.memory_capacity, size=self.batch_size)
		return self.memory[indices, :]

	def learn(self):
		# update target net 学习一段时间更新目标网络
		if self.update_target_step % self.target_replace_iter == 0:
			self.target_net.load_state_dict(self.eval_net.state_dict())
		self.update_target_step += 1

		# batch data
		bm = self.sample()
		bs = torch.FloatTensor(bm[:, :self.state_dim]).to(device)
		ba = torch.LongTensor(bm[:, self.state_dim:self.state_dim + 1]).to(device)
		br = torch.FloatTensor(bm[:, -self.state_dim - 1: -self.state_dim]).to(device)
		bs_ = torch.FloatTensor(bm[:, -self.state_dim:]).to(device)

		q_eval = self.eval_net(bs).gather(1, ba)

		q_next = self.target_net(bs_).detach()

		q_target = br + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)

		loss = self.loss_func(q_eval, q_target)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()


