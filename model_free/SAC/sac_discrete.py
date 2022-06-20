import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
import collections
import random
from utils.replaybuffer import ReplayBuffer


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim_1, hidden_dim_2):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim_1)
        self.fc2 = torch.nn.Linear(hidden_dim_1, action_dim)
        # self.fc3 = torch.nn.Linear(hidden_dim_2, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return F.softmax(self.fc2(x), dim=1)


class QValueNet(torch.nn.Module):
    """ 两层MLP """
    def __init__(self, state_dim, action_dim, hidden_dim_1, hidden_dim_2):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim_1)
        self.fc2 = torch.nn.Linear(hidden_dim_1, action_dim)
        # self.fc3 = torch.nn.Linear(hidden_dim_2, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return self.fc2(x)


class SAC_Discrete:
    ''' 处理离散动作的SAC算法 '''
    def __init__(self, state_dim, action_dim, config, device):

        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.lr_alpha = config["lr_alpha"]
        self.lr_actor = config["lr_actor"]
        self.lr_critic = config["lr_critic"]
        self.batch_size = config["batch_size"]
        self.buffer_size = config["buffer_size"]
        self.hidden_dim_1 = config["hidden_dim_1"]
        self.hidden_dim_2 = config["hidden_dim_2"]
        self.buffer = ReplayBuffer(self.buffer_size, self.batch_size, device)
        self.target_entropy = config["target_entropy"]  # 目标熵的大小
        self.device = device
        # 策略网络
        self.actor = PolicyNet(state_dim, action_dim, self.hidden_dim_1, self.hidden_dim_2).to(device)
        # 第一个Q网络
        self.critic_1 = QValueNet(state_dim, action_dim, self.hidden_dim_1, self.hidden_dim_2).to(device)
        # 第二个Q网络
        self.critic_2 = QValueNet(state_dim, action_dim, self.hidden_dim_1, self.hidden_dim_2).to(device)
        self.target_critic_1 = QValueNet(state_dim, action_dim, self.hidden_dim_1, self.hidden_dim_2).to(device)
        self.target_critic_2 = QValueNet(state_dim, action_dim, self.hidden_dim_1, self.hidden_dim_2).to(device)
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr_actor)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=self.lr_critic)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=self.lr_critic)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr_alpha)

    def select_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self):
        states, actions, rewards, next_states, dones = self.buffer.sample()

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_q_values = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(
            F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(
            F.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        # 直接根据概率计算熵
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)  # 直接根据概率计算期望
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
        return actor_loss.item()
