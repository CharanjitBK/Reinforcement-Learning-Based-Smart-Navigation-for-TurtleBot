import numpy as np
import torch
import torch.nn as nn

from ..common.settings import POLICY_NOISE_CLIP
from .off_policy_agent import OffPolicyAgent, Network

LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6


# ==================== NETWORKS ==================== #

class Actor(Network):
    """
    Gaussian Policy Actor (outputs mean + log_std for SAC)
    """
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Actor, self).__init__(name)

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std = nn.Linear(hidden_size, action_size)

        self.apply(super().init_weights)

    def forward(self, states, visualize=False):
        x = torch.relu(self.fc1(states))
        x = torch.relu(self.fc2(x))

        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        if visualize and self.visual:
            self.visual.update_layers(states, mu, [x], [self.fc1.bias, self.fc2.bias])

        return mu, log_std

    def sample(self, states):
        mu, log_std = self.forward(states)
        std = log_std.exp()

        noise = torch.randn_like(mu)
        action = mu + noise * std

        action_tanh = torch.tanh(action)

        log_prob = -0.5 * (((action - mu) / (std + EPSILON)) ** 2 +
                           2 * log_std + np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - action_tanh.pow(2) + EPSILON).sum(dim=-1, keepdim=True)

        return action_tanh, log_prob


class CriticQ(Network):
    """
    Two Q networks used by SAC
    """
    def __init__(self, name, state_size, action_size, hidden_size):
        super(CriticQ, self).__init__(name)

        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q1_out = nn.Linear(hidden_size, 1)

        self.fc3 = nn.Linear(state_size + action_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.q2_out = nn.Linear(hidden_size, 1)

        self.apply(super().init_weights)

    def forward(self, states, actions):
        sa = torch.cat([states, actions], dim=1)

        x1 = torch.relu(self.fc1(sa))
        x1 = torch.relu(self.fc2(x1))
        q1 = self.q1_out(x1)

        x2 = torch.relu(self.fc3(sa))
        x2 = torch.relu(self.fc4(x2))
        q2 = self.q2_out(x2)

        return q1, q2


class ValueNet(Network):
    """
    Critic V network for SAC
    """
    def __init__(self, name, state_size, action_size,hidden_size):
        super(ValueNet, self).__init__(name)

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.v_out = nn.Linear(hidden_size, 1)

        self.apply(super().init_weights)

    def forward(self, states):
        x = torch.relu(self.fc1(states))
        x = torch.relu(self.fc2(x))
        return self.v_out(x)


# ==================== SAC AGENT ==================== #

class SAC(OffPolicyAgent):
    """
    Soft Actor-Critic compatible with the DRL framework
    """
    def __init__(self, device, sim_speed):
        super().__init__(device, sim_speed)

        # temperature (entropy)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.target_entropy = -float(self.action_size)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        # networks
        self.actor = self.create_network(Actor, 'actor')
        self.actor_optimizer = self.create_optimizer(self.actor)

        self.critic_q = self.create_network(CriticQ, 'critic_q')
        self.critic_q_target = self.create_network(CriticQ, 'critic_q_target')
        self.critic_q_optimizer = self.create_optimizer(self.critic_q)

        self.value = self.create_network(ValueNet, 'value')
        self.value_target = self.create_network(ValueNet, 'value_target')
        self.value_optimizer = self.create_optimizer(self.value)

        self.hard_update(self.critic_q_target, self.critic_q)
        self.hard_update(self.value_target, self.value)

        self.last_actor_loss = 0

    # -------- get action -------- #
    # def get_action(self, state, is_training, step, visualize=False):
    #     state = torch.from_numpy(np.asarray(state, np.float32)).to(self.device)
    #     action, _ = self.actor.sample(state)

    #     if not is_training:
    #         action = torch.tanh(self.actor.forward(state)[0])

    #     return action.detach().cpu().numpy().tolist()

    def get_action(self, state, is_training, step, visualize=False):
        # Convert state to tensor
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            mu, log_std = self.actor.forward(state)

            if is_training:
                std = log_std.exp()
                noise = torch.randn_like(mu)
                action = mu + noise * std
            else:
                action = mu  # deterministic for testing

        # Only apply tanh once
        action = torch.tanh(action).squeeze(0).cpu().numpy()

        # -------------------------
        # TurtleBot3 scaling
        # -------------------------
        # Linear velocity: map [-1, 1] -> [0, 0.22]
        linear = float(np.clip((action[0] + 1) / 2 * 0.22, 0.01, 0.22))
        # Angular velocity: map [-1, 1] -> [-2.84, 2.84]
        angular = float(np.clip(action[1] * 2.84, -2.84, 2.84))

        return [linear, angular]




    def get_action_random(self):
        return [np.random.uniform(-1.0, 1.0)] * self.action_size

    # -------- training -------- #
    def train(self, state, action, reward, next_state, done):
        alpha = self.log_alpha.exp()

        # ----- VALUE -----
        with torch.no_grad():
            next_action, next_logp = self.actor.sample(next_state)
            q1_next, q2_next = self.critic_q_target(next_state, next_action)
            q_min = torch.min(q1_next, q2_next)
            v_target = q_min - alpha * next_logp

        v = self.value(state)
        loss_value = self.loss_function(v, v_target)
        self.value_optimizer.zero_grad()
        loss_value.backward()
        self.value_optimizer.step()

        # ----- Q CRITICS -----
        with torch.no_grad():
            v_next = self.value_target(next_state)
            q_target = reward + (1 - done) * self.discount_factor * v_next

        q1, q2 = self.critic_q(state, action)
        loss_q = self.loss_function(q1, q_target) + self.loss_function(q2, q_target)
        self.critic_q_optimizer.zero_grad()
        loss_q.backward()
        self.critic_q_optimizer.step()

        # ----- ACTOR -----
        new_action, logp = self.actor.sample(state)
        q1_new, q2_new = self.critic_q(state, new_action)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (alpha * logp - q_new).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.last_actor_loss = actor_loss.item()

        # ----- ENTROPY TEMPERATURE -----
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ----- TARGET NETS -----
        self.soft_update(self.value_target, self.value,self.tau)
        self.soft_update(self.critic_q_target, self.critic_q,self.tau)

        return [loss_q.mean().detach().cpu(), self.last_actor_loss]