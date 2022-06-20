# -*-coding:utf-8-*-
# @Time  : 2022/3/1 22:44
# @Author: https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py
# @File  : ppo.py
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

################################## set self.device ##################################
print("============================================================================================")
# set self.device to cpu or cuda
# self.device = torch.self.device('cpu')
# if (torch.cuda.is_available()):
# 	self.device = torch.self.device('cuda:0')
# 	torch.cuda.empty_cache()
# 	print("self.device set to : " + str(torch.cuda.get_self.device_name(self.device)))
# else:
# 	print("self.device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
	def __init__(self):
		self.actions = []
		self.states = []
		self.logprobs = []
		self.rewards = []
		self.is_terminals = []

	def clear(self):
		del self.actions[:]
		del self.states[:]
		del self.logprobs[:]
		del self.rewards[:]
		del self.is_terminals[:]

	def add(self, state, action, reward, state_, done):
		self.is_terminals.append(done)
		self.rewards.append(reward)


class ActorCritic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim_1, hidden_dim_2, has_continuous_action_space, action_std_init):
		super(ActorCritic, self).__init__()

		self.has_continuous_action_space = has_continuous_action_space

		if has_continuous_action_space:
			self.action_dim = action_dim
			self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)
		# actor
		if has_continuous_action_space:
			self.actor = nn.Sequential(
				nn.Linear(state_dim, hidden_dim_1),
				nn.Tanh(),
				nn.Linear(hidden_dim_1, hidden_dim_2),
				nn.Tanh(),
				nn.Linear(hidden_dim_2, action_dim),
			)
		else:
			self.actor = nn.Sequential(
				nn.Linear(state_dim, hidden_dim_1),
				nn.Tanh(),
				nn.Linear(hidden_dim_1, hidden_dim_2),
				nn.Tanh(),
				nn.Linear(hidden_dim_2, action_dim),
				nn.Softmax(dim=-1)
			)
		# critic
		self.critic = nn.Sequential(
			nn.Linear(state_dim, hidden_dim_1),
			nn.Tanh(),
			nn.Linear(hidden_dim_1, hidden_dim_2),
			nn.Tanh(),
			nn.Linear(hidden_dim_2, 1)
		)

	def set_action_std(self, new_action_std):
		if self.has_continuous_action_space:
			self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
		else:
			print("--------------------------------------------------------------------------------------------")
			print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
			print("--------------------------------------------------------------------------------------------")

	def forward(self):
		raise NotImplementedError

	def act(self, state):
		if self.has_continuous_action_space:
			action_mean = self.actor(state)
			cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
			dist = MultivariateNormal(action_mean, cov_mat)
		else:
			action_probs = self.actor(state)
			dist = Categorical(action_probs)

		action = dist.sample()
		action_logprob = dist.log_prob(action)

		return action.detach(), action_logprob.detach()

	def evaluate(self, state, action):

		if self.has_continuous_action_space:
			action_mean = self.actor(state)

			action_var = self.action_var.expand_as(action_mean)
			cov_mat = torch.diag_embed(action_var).to(self.device)
			dist = MultivariateNormal(action_mean, cov_mat)

			# For Single Action Environments.
			if self.action_dim == 1:
				action = action.reshape(-1, self.action_dim)
		else:
			action_probs = self.actor(state)
			dist = Categorical(action_probs)
		action_logprobs = dist.log_prob(action)
		dist_entropy = dist.entropy()
		state_values = self.critic(state)

		return action_logprobs, state_values, dist_entropy


class PPO:
	def __init__(self, state_dim, action_dim, config, device):

		self.has_continuous_action_space = config["has_continuous_action_space"]

		if self.has_continuous_action_space:
			self.action_std = config["action_std_init"]

		self.gamma = config["gamma"]
		self.eps_clip = config["eps_clip"]
		self.K_epochs = config["K_epochs"]
		self.action_std_init = config["action_std_init"]
		self.action_std = config["action_std"]
		self.action_std_decay_rate = config["action_std_decay_rate"]
		self.action_std_decay_freq = config["action_std_decay_freq"]
		self.min_action_std = config["min_action_std"]

		self.lr_actor = config["lr_actor"]
		self.lr_critic = config["lr_critic"]
		self.hidden_dim_1 = config["hidden_dim_1"]
		self.hidden_dim_2 = config["hidden_dim_2"]

		self.device = device

		self.buffer = RolloutBuffer()

		self.policy = ActorCritic(state_dim, action_dim, self.hidden_dim_1, self.hidden_dim_2,
								self.has_continuous_action_space, self.action_std_init).to(self.device)
		self.optimizer = torch.optim.Adam([
			{'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
			{'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
		])

		self.policy_old = ActorCritic(state_dim, action_dim, self.hidden_dim_1, self.hidden_dim_2,
									self.has_continuous_action_space, self.action_std_init).to(self.device)
		self.policy_old.load_state_dict(self.policy.state_dict())

		self.MseLoss = nn.MSELoss()
		self.timestep = 0

	def set_action_std(self, new_action_std):
		if self.has_continuous_action_space:
			self.action_std = new_action_std
			self.policy.set_action_std(new_action_std)
			self.policy_old.set_action_std(new_action_std)
		else:
			print("--------------------------------------------------------------------------------------------")
			print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
			print("--------------------------------------------------------------------------------------------")

	def decay_action_std(self,):
		print("--------------------------------------------------------------------------------------------")
		if self.has_continuous_action_space:
			self.action_std = self.action_std - self.action_std_decay_rate
			self.action_std = round(self.action_std, 4)
			if self.action_std <= self.min_action_std:
				self.action_std = self.min_action_std
				print("setting actor output action_std to min_action_std : ", self.action_std)
			else:
				print("setting actor output action_std to : ", self.action_std)
			self.set_action_std(self.action_std)

		else:
			print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
		print("--------------------------------------------------------------------------------------------")

	def select_action(self, state):
		self.timestep += 1
		if self.has_continuous_action_space:
			with torch.no_grad():
				state = torch.FloatTensor(state).to(self.device)
				action, action_logprob = self.policy_old.act(state)

			self.buffer.states.append(state)
			self.buffer.actions.append(action)
			self.buffer.logprobs.append(action_logprob)

			# decay action std
			if self.timestep % self.action_std_decay_freq == 0:
				self.decay_action_std()

			return action.detach().cpu().numpy().flatten()

		else:
			with torch.no_grad():
				state = torch.FloatTensor(state).to(self.device)
				action, action_logprob = self.policy_old.act(state)

			self.buffer.states.append(state)
			self.buffer.actions.append(action)
			self.buffer.logprobs.append(action_logprob)

			return action.item()

	def update(self):
		# Monte Carlo estimate of returns
		rewards = []
		discounted_reward = 0
		for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
			if is_terminal:
				discounted_reward = 0
			discounted_reward = reward + (self.gamma * discounted_reward)
			rewards.insert(0, discounted_reward)

		# Normalizing the rewards
		rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
		rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

		# convert list to tensor
		old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
		old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
		old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

		loss_k = []
		# Optimize policy for K epochs
		for _ in range(self.K_epochs):
			# Evaluating old actions and values
			logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

			# match state_values tensor dimensions with rewards tensor
			state_values = torch.squeeze(state_values)

			# Finding the ratio (pi_theta / pi_theta__old)
			ratios = torch.exp(logprobs - old_logprobs.detach())

			# Finding Surrogate Loss
			advantages = rewards - state_values.detach()
			surr1 = ratios * advantages
			surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

			# final loss of clipped objective PPO
			loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

			loss_ = loss.clone().mean().data if self.device == torch.device(
				'cpu') else loss.clone().mean().data.cpu()
			loss_k.append(loss_)

			# take gradient step
			self.optimizer.zero_grad()
			loss.mean().backward()
			self.optimizer.step()

		# Copy new weights into old policy
		self.policy_old.load_state_dict(self.policy.state_dict())

		# clear buffer
		self.buffer.clear()

		return sum(loss_k)/len(loss_k)

	def save(self, checkpoint_path):
		torch.save(self.policy_old.state_dict(), checkpoint_path)

	def load(self, checkpoint_path):
		self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
		self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))