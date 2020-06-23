import copy
import random
import pickle
from collections import namedtuple, deque, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from model import QNetwork

BUFFER_SIZE = int(2e5)  # replay buffer size
BATCH_SIZE = 512  # minibatch size
GAMMA = 0.998  # discount factor 0.99
TAU = 1e-3  # for soft update of target parameters
LR = 0.5e-4  # learning rate 0.5e-4 works
UPDATE_EVERY = 10 # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, double_dqn=True):
        self.state_size = state_size
        self.action_size = action_size
        self.double_dqn = double_dqn

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.qnetwork_target = copy.deepcopy(self.qnetwork_local)
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        self.t_step = 0

    def save(self, path, *data):
        torch.save(self.qnetwork_local.state_dict(), path / "model_checkpoint.local")
        torch.save(self.qnetwork_target.state_dict(), path / "model_checkpoint.target")
        torch.save(self.optimizer.state_dict(), path / 'model_checkpoint.optimizer')
        with open(path / 'model_checkpoint.meta', 'wb') as file:
            pickle.dump(data, file)

    def load(self, path, *defaults):
        try:
            print("Loading model from checkpoint...")
            self.qnetwork_local.load_state_dict(torch.load(path / 'model_checkpoint.local'))
            self.qnetwork_target.load_state_dict(torch.load(path / 'model_checkpoint.target'))
            self.optimizer.load_state_dict(torch.load(path / 'model_checkpoint.optimizer'))
            with open(path / 'model_checkpoint.meta', 'rb') as file:
                return pickle.load(file)
        except:
            print("No checkpoint file was found")
            return defaults

    def step(self, state, action, reward, next_state, done, train=True):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if train and len(self.memory) > BATCH_SIZE and self.t_step == 0:
            self.learn(self.memory.sample(), GAMMA)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
              return np.argmax(action_values.cpu().data.numpy())
        else: return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        if self.double_dqn:
              Q_best_action = self.qnetwork_local(next_states).max(1)[1]
              Q_targets_next = self.qnetwork_target(next_states).gather(1, Q_best_action.unsqueeze(-1))
        else: Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(-1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute loss and perform a gradient step
        self.optimizer.zero_grad()
        loss = F.mse_loss(Q_expected, Q_targets)
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)



Transition = namedtuple("Experience", ("state", "action", "reward", "next_state", "done"))

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size):
        self.action_size = action_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.memory.append(Transition(np.expand_dims(state, 0), action, reward, np.expand_dims(next_state, 0), done))

        # if len(self.memory) < self.buffer_size:
        #     self.memory.append(Transition(np.expand_dims(state, 0), action, reward, np.expand_dims(next_state, 0), done))
        # else:
        #     position = np.random.randint(self.buffer_size) # max(np.random.randint(self.buffer_size), np.random.randint(self.buffer_size))
        #     self.memory[position] = Transition(np.expand_dims(state, 0), action, reward, np.expand_dims(next_state, 0), done)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states      = torch.from_numpy(self.stack([e.state      for e in experiences if e is not None])).float().to(device)
        actions     = torch.from_numpy(self.stack([e.action     for e in experiences if e is not None])).long().to(device)
        rewards     = torch.from_numpy(self.stack([e.reward     for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(self.stack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones       = torch.from_numpy(self.stack([e.done       for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def stack(self, states):
        # sub_dim = len(states[0][0]) if isinstance(states[0], Iterable) else 1
        # np_states = np.reshape(np.array(states), (len(states), sub_dim))
        sub_dims = states[0].shape[1:] if isinstance(states[0], Iterable) else [1]
        return np.reshape(np.array(states), (len(states), *sub_dims))

    def __len__(self):
        return len(self.memory)
