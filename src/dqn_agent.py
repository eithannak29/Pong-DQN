import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from model import DQN

class DQNAgent:
    def __init__(self, input_shape, num_actions, actions, lr=1e-4, gamma=0.99, batch_size=32, buffer_size=10000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(input_shape, num_actions).to(self.device)
        self.target_model = DQN(input_shape, num_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=buffer_size)
        self.num_actions = num_actions
        self.actions = actions  # Action mapping

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            action_idx = random.randint(0, self.num_actions - 1)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state)
            action_idx = q_values.argmax().item()
        action = self.actions[action_idx]
        return action_idx, action

    def store_transition(self, transition):
        self.replay_buffer.append(transition)

    def sample_memory(self):
        return random.sample(self.replay_buffer, self.batch_size)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.sample_memory()
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
