"""
 Soft-Q Leaning (Actor_Critic), Reinforcement Learning.
 torch实现 Soft-Q Leaning算法,SQL
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.replaybuffer import ReplayBuffer
import numpy as np
class SoftQNetwork(nn.Module):
  def __init__(self, state_dim, hidden_dim, action_dim, alpha):
    super(SoftQNetwork, self).__init__()
    self.layer = nn.Sequential(
				nn.Linear(state_dim, hidden_dim),
				nn.ReLU(),
        nn.Linear(hidden_dim, action_dim)
		)
    self.alpha = alpha
  
  def get_Q(self, state):
    Q = self.layer(state)
    return Q
  
  def get_V(self, Q):
    # 通过importance sampling转成期望计算，通过采样取平均即可
    # V_soft(s) = α * log(E_q[exp(Q_soft(s, a) / α) / q ] )
    V = self.alpha * torch.logsumexp(Q / self.alpha, dim = -1)
    return V

class SoftQLearning:
  def __init__(self, state_dim, action_dim, config, device):
    self.device = device
    self.state_dim = state_dim
    self.alpha = 2
    self.gamma = config['gamma']
    self.soft_q_net = SoftQNetwork(state_dim, config['hidden_dim_2'], action_dim, self.alpha).to(self.device)
    self.q_criterion = nn.MSELoss()
    self.v_criterion = nn.MSELoss()
    self.soft_q_optimizer = torch.optim.Adam(self.soft_q_net.parameters(), lr=config['lr_alpha'])

    # Experience Memory Replay
    self.buffer = ReplayBuffer(config['buffer_size'], config['batch_size'], device)
  def select_action(self, state):
    state = torch.FloatTensor(state).to(self.device)
    Q = self.soft_q_net.get_Q(state)
    # π(a|s) = exp(Q_soft(s,a) - V(s) / α)
    dist = torch.exp((Q - self.soft_q_net.get_V(Q)) / self.alpha)
    # uniform sampling
    dist = dist / torch.sum(dist)
    # 从Categorical分布中采样动作
    m = torch.distributions.Categorical(dist.squeeze(0))
    a = m.sample()
    return a.item()
  
  def update(self):
    # 这里的代码跟论文有点不一样，因为这里的没有用action sample net，因此只需要更新一个参数
    # 从memory中采样一批数据，经验回放
    states, actions, rewards, states_, terminals = self.buffer.sample()
    # 计算Q_soft(s_t,a_t) = r + gamma * E[V_soft(s_{t+1})]
    Q = self.soft_q_net.get_Q(states).squeeze(1)
    Est_Q = Q.clone()
    next_Q = self.soft_q_net.get_Q(states_).squeeze(1)
    next_V = self.soft_q_net.get_V(next_Q)
    for i in range(len(actions)):
       Est_Q[i][actions[i]] = rewards[i] + self.gamma * next_V[i]
    
    # 通过对MSE-Loss进行随机梯度下降进行优化
    # 即J_Q(theta) = E[(\hat(Q_soft(s_t, a_t))-Q_soft(s_t,a_t))^2 / 2]
    Q_loss = F.mse_loss(Q, Est_Q.detach())
    self.soft_q_optimizer.zero_grad()
    Q_loss.backward()
    self.soft_q_optimizer.step()

    return Q_loss
    

