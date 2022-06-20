# -*-coding:utf-8-*-
# @Time  : 2022/4/22 22:06
# @Author: https://github.com/BY571/SAC_discrete
# @File  : sac_2.py

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical
import numpy as np
import copy
from utils.replaybuffer import ReplayBuffer


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_dim_1, hidden_dim_2):
        """Initialize parameters and build model.
        Params
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, action_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = self.softmax(self.fc3(x))
        return action_probs

    def evaluate(self, state, epsilon=1e-6):
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities

    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities

    def get_det_action(self, state, device):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample().to(device)
        return action.detach().cpu().numpy()


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_dim_1, hidden_dim_2, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers
        """
        super(Critic, self).__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SAC_Discrete(nn.Module):
    """Interacts with and learns from the environment."""

    def __init__(self, state_dim, action_dim, config, device):
        """Initialize an Agent object.

        Params
        ======
            state_dim (int): dimension of each state
            action_dim (int): dimension of each action
            config (int): config
        """
        super(SAC_Discrete, self).__init__()
        self.name = "SAC_Discrete"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.hidden_dim_1 = config["hidden_dim_1"]
        self.hidden_dim_2 = config["hidden_dim_2"]

        self.clip_grad_param = config["clip_grad_param"]
        self.lr_alpha = config["lr_alpha"]
        self.lr_actor = config["lr_actor"]
        self.lr_critic = config["lr_critic"]
        # buffer
        self.buffer = ReplayBuffer(buffer_size=config["buffer_size"], batch_size=config["batch_size"], device=device)
        self.target_entropy = -action_dim  # -dim(A)

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=self.lr_alpha)

        # Actor Network

        self.actor_local = Actor(state_dim, action_dim, self.hidden_dim_1, self.hidden_dim_2).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network)

        self.critic1 = Critic(state_dim, action_dim, self.hidden_dim_1, self.hidden_dim_2, 2).to(device)
        self.critic2 = Critic(state_dim, action_dim, self.hidden_dim_1, self.hidden_dim_2, 1).to(device)

        assert self.critic1.parameters() != self.critic2.parameters()

        self.critic1_target = Critic(state_dim, action_dim, self.hidden_dim_1, self.hidden_dim_2, 2).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_dim, action_dim, self.hidden_dim_1, self.hidden_dim_2, 1).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr_critic)

    def select_action(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)

        with torch.no_grad():
            action = self.actor_local.get_det_action(state, self.device)
        return action

    def calc_policy_loss(self, states, alpha):
        _, action_probs, log_pis = self.actor_local.evaluate(states)

        q1 = self.critic1(states)
        q2 = self.critic2(states)
        min_Q = torch.min(q1, q2)
        actor_loss = (action_probs * (alpha * log_pis - min_Q)).sum(1).mean()
        log_action_pi = torch.sum(log_pis * action_probs, dim=1)
        return actor_loss, log_action_pi

    def update(self, transition_dict=None):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        experiences = self.buffer.sample()
        states, actions, rewards, next_states, dones = experiences
        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self.calc_policy_loss(states, current_alpha.to(self.device))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Compute alpha loss
        alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            _, action_probs, log_pis = self.actor_local.evaluate(next_states)
            Q_target1_next = self.critic1_target(next_states)
            Q_target2_next = self.critic2_target(next_states)
            Q_target_next = action_probs * (
                    torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * log_pis)

            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (self.gamma * (1 - dones) * Q_target_next.sum(dim=1).unsqueeze(-1))

        # Compute critic loss
        q1 = self.critic1(states).gather(1, actions)
        q2 = self.critic2(states).gather(1, actions)

        # critic1_loss = torch.mean(F.mse_loss(q1, Q_targets))
        # critic2_loss = torch.mean(F.mse_loss(q2, Q_targets))
        critic1_loss = 0.5 * F.mse_loss(q1, Q_targets)
        critic2_loss = 0.5 * F.mse_loss(q2, Q_targets)

        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        policy_loss, alpha_loss, bellmann_error1, bellmann_error2 = actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item()
        return policy_loss

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
