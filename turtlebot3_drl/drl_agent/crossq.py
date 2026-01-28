import numpy as np
import torch
import torch.nn as nn

from ..common.settings import CROSS_HIDDEN
from .off_policy_agent import OffPolicyAgent, Network

LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6


# ==================== NETWORKS ==================== #

class Actor(Network):
    def __init__(self, name, state_size, action_size, hidden_size=CROSS_HIDDEN):
        super().__init__(name)

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size, momentum=0.01)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size, momentum=0.01)

        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std = nn.Linear(hidden_size, action_size)

        self.apply(super().init_weights)
        self._init_bn()

    def _init_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, states,visualize=False):
        x = torch.relu(self.bn1(self.fc1(states)))
        x = torch.relu(self.bn2(self.fc2(x)))
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        if visualize and self.visual:
            self.visual.update_layers(states, mu, [x], [self.fc1.bias, self.fc2.bias])
        return mu, log_std

    def sample(self, states):
        mu, log_std = self.forward(states)
        std = log_std.exp()
        noise = torch.randn_like(mu)

        action = mu + noise * std
        action_tanh = torch.tanh(action)

        log_prob = -0.5 * (
            ((action - mu) / (std + EPSILON)) ** 2
            + 2 * log_std
            + np.log(2 * np.pi)
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - action_tanh.pow(2) + EPSILON).sum(dim=-1, keepdim=True)

        return action_tanh, log_prob


class CriticQ(Network):
    """
    CrossQ critic with joint (s,a) / (s',a') BatchNorm forward
    """
    def __init__(self, name, state_size, action_size, hidden_size=CROSS_HIDDEN):
        super().__init__(name)

        # Q1
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size, momentum=0.01)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size, momentum=0.01)
        self.q1_out = nn.Linear(hidden_size, 1)

        # Q2
        self.fc3 = nn.Linear(state_size + action_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size, momentum=0.01)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size, momentum=0.01)
        self.q2_out = nn.Linear(hidden_size, 1)

        self.apply(super().init_weights)
        self._init_bn()

    def _init_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, states, actions):
        sa = torch.cat([states, actions], dim=1)

        x1 = torch.relu(self.bn1(self.fc1(sa)))
        x1 = torch.relu(self.bn2(self.fc2(x1)))
        q1 = self.q1_out(x1)

        x2 = torch.relu(self.bn3(self.fc3(sa)))
        x2 = torch.relu(self.bn4(self.fc4(x2)))
        q2 = self.q2_out(x2)

        return q1, q2


# ==================== CROSSQ AGENT ==================== #

class CrossQ(OffPolicyAgent):
    def __init__(self, device, sim_speed):
        super().__init__(device, sim_speed)

        # Entropy temperature
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.target_entropy = -float(self.action_size)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        # Networks
        self.actor = self.create_network(Actor, "actor")
        self.actor_optimizer = self.create_optimizer(self.actor)

        self.critic_q = self.create_network(CriticQ, "critic_q")
        self.critic_q_optimizer = self.create_optimizer(self.critic_q)

        self.policy_delay = 2
        self.total_it = 0
        self.last_actor_loss = 0.0

    # ---------- ACTION ---------- #

    def get_action(self, state, is_training, step,visualize=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        self.actor.eval()
        with torch.no_grad():
            if is_training:
                action, _ = self.actor.sample(state)
            else:
                mu, _ = self.actor(state)
                action = torch.tanh(mu)
        self.actor.train()

        return action.detach().cpu().numpy()[0].tolist()

    def get_action_random(self):
        return np.random.uniform(-1.0, 1.0, size=self.action_size).tolist()

    # ---------- TRAIN ---------- #

    def train(self, state, action, reward, next_state, done):
        self.total_it += 1
        alpha = self.log_alpha.exp()

        # -------- Critic Update -------- #

        with torch.no_grad():
            next_action, next_logp = self.actor.sample(next_state)

        # Cross-batch BN forward
        states_cat = torch.cat([state, next_state], dim=0)
        actions_cat = torch.cat([action, next_action], dim=0)

        q1_cat, q2_cat = self.critic_q(states_cat, actions_cat)
        q1, q1_next = torch.chunk(q1_cat, 2, dim=0)
        q2, q2_next = torch.chunk(q2_cat, 2, dim=0)

        with torch.no_grad():
            q_next = torch.min(q1_next, q2_next)
            q_target = reward + (1 - done) * self.discount_factor * (q_next - alpha * next_logp)

        critic_loss = (
            self.loss_function(q1, q_target) +
            self.loss_function(q2, q_target)
        )

        self.critic_q_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_q_optimizer.step()

        # -------- Delayed Actor + Alpha Update -------- #

        if self.total_it % self.policy_delay == 0:
            for p in self.critic_q.parameters():
                p.requires_grad = False

            self.critic_q.eval()

            new_action, logp = self.actor.sample(state)
            q1_pi, q2_pi = self.critic_q(state, new_action)
            q_pi = torch.min(q1_pi, q2_pi)

            actor_loss = (alpha * logp - q_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.last_actor_loss = actor_loss.item()

            # Temperature update
            entropy = -logp.detach()
            alpha_loss = (self.log_alpha * (entropy - self.target_entropy)).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.critic_q.train()
            for p in self.critic_q.parameters():
                p.requires_grad = True

        return [
            critic_loss.mean().detach().cpu(),
            self.last_actor_loss,
        ]
