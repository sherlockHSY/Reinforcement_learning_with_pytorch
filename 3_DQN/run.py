# -*-coding:utf-8-*-
# @Time  : 2022/3/8 21:31
# @Author: hsy
# @File  : run.py
import gym
import torch
import numpy as np
from my_dqn import DQN


if __name__ == '__main__':
	env = gym.make('CartPole-v0').unwrapped
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.n

	# hyper parameter
	batch_size = 32
	lr = 0.01                                       # learning rate
	epsilon = 0.9                                   # greedy policy
	gamma = 0.9                                     # reward discount
	target_replace_iter = 100                       # update
	memory_capacity = 2000
	max_ep_len = 200  # max time steps in one episode
	max_training_timesteps = int(4e2)  # break training loop if time steps > max_training_time steps
	random_seed = 10

	# Random seed
	torch.manual_seed(random_seed)
	env.seed(random_seed)
	np.random.seed(random_seed)

	print("state space dimension : ", state_dim)
	print("action space dimension : ", action_dim)

	dqn = DQN(state_dim=state_dim,
			  action_dim=action_dim,
			  batch_size=batch_size,
			  lr=lr,
			  epsilon=epsilon,
			  gamma=gamma,
			  memory_capacity=memory_capacity,
			  taget_replace_iter=target_replace_iter)

	time_step = 0
	i_episode = 0
	learn_flag = 0
	for i in range(400):
		state = env.reset()
		current_ep_reward = 0

		while True:
			env.render()

			action = dqn.choose_action(state)

			state_, reward, done, info = env.step(action)

			# 修改奖励 (不修改也可以，修改奖励只是为了更快地得到训练好的摆杆)
			x, x_dot, theta, theta_dot = state_
			r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
			r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
			new_reward = r1 + r2

			dqn.store_transition(state, action, new_reward, state_)
			current_ep_reward += new_reward

			state = state_

			if dqn.pointer > dqn.memory_capacity:
				if learn_flag == 0:
					print('begin learn')
					learn_flag += 1
				dqn.learn()
			if done:
				print("Episode: ", i, "  reward: ", round(current_ep_reward, 2))
				break