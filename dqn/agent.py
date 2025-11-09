import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .replay_buffer import ReplayBuffer

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=(128,128)):
        super().__init__()
        layers = []
        last = state_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, action_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=3e-4, tau=0.005,
                 buffer_capacity=100000, batch_size=128, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.q = QNet(state_dim, action_dim).to(self.device)
        self.q_target = QNet(state_dim, action_dim).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optim = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(state_dim, capacity=buffer_capacity, batch_size=batch_size)
        self.action_dim = action_dim
        self._loss_fn = nn.SmoothL1Loss()

        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay_steps = 50000
        self.total_steps = 0

    def act(self, state):
        eps = self._epsilon()
        if np.random.rand() < eps:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q = self.q(s)
            return int(torch.argmax(q, dim=1).item())

    def _epsilon(self):
        # Linear decay
        t = min(1.0, self.total_steps / max(1, self.eps_decay_steps))
        return self.eps_start + t * (self.eps_end - self.eps_start)

    def push(self, *transition):
        self.buffer.push(*transition)

    def train_step(self):
        if not self.buffer.can_sample():
            return None
        states, actions, rewards, next_states, dones = self.buffer.sample()
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().unsqueeze(-1).to(self.device)
        rewards = torch.from_numpy(rewards).float().unsqueeze(-1).to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().unsqueeze(-1).to(self.device)

        # Q(s,a)
        q_values = self.q(states).gather(1, actions)

        # target: r + gamma * max_a' Q_target(s',a')
        with torch.no_grad():
            max_next_q = self.q_target(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + (1.0 - dones) * self.gamma * max_next_q

        loss = self._loss_fn(q_values, target)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.optim.step()

        # soft update target
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
        return loss.item()

    def save(self, path):
        torch.save({
            "q": self.q.state_dict(),
        }, path)

    def load(self, path, map_location=None):
        data = torch.load(path, map_location=map_location or self.device)
        self.q.load_state_dict(data["q"])
        self.q_target.load_state_dict(self.q.state_dict())
