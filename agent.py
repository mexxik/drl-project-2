import math
import copy
import random
import torch
import numpy as np
from collections import deque
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Tensor = torch.Tensor


class Parameters(object):
    def __init__(self):
        self.num_episodes = 2000
        self.solve_score = 30

        self.replay_capacity = 50000
        self.batch_size = 128

        self.gamma = 0.99

        self.learning_rate_actor = 0.001
        self.learning_rate_critic = 0.001

        self.weight_decay = 0

        self.tau = 0.001

        self.epsilon = 1.0
        self.epsilon_decay = 0.000001


class ExperienceReplay(object):
    def __init__(self, params):
        self.params = params

        self.memory = deque(maxlen=self.params.replay_capacity)
        self.position = 0

    def push(self, state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)

        #if self.position >= len(self.memory):
        self.memory.append(transition)
        #else:
        #    self.memory[self.position] = transition

        #self.position = (self.position + 1) % self.params.replay_capacity

    def sample(self):
        return zip(*random.sample(self.memory, self.params.batch_size))

    def __len__(self):
        return len(self.memory)


class ActorNetwork(torch.nn.Module):
    def __init__(self, params, num_states, num_actions):
        super(ActorNetwork, self).__init__()

        self.params = params
        self.num_states = num_states
        self.num_actions = num_actions

        self.fc1_size = 400
        self.fc2_size = 300

        self.fc1 = torch.nn.Linear(num_states, self.fc1_size)
        self.fc2 = torch.nn.Linear(self.fc1_size, self.fc2_size)
        self.fc3 = torch.nn.Linear(self.fc2_size, self.num_actions)

        self.reset_parameters()

    def _init_hidden(self, layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)

        return -lim, lim

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*self._init_hidden(self.fc1))
        self.fc2.weight.data.uniform_(*self._init_hidden(self.fc2))
        self.fc3.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        return torch.tanh(self.fc3(x))


class CriticNetwork(torch.nn.Module):
    def __init__(self, params, num_states, num_actions):
        super(CriticNetwork, self).__init__()

        self.params = params
        self.num_states = num_states
        self.num_actions = num_actions

        self.fc1_size = 400
        self.fc2_size = 300

        self.fc1 = torch.nn.Linear(num_states, self.fc1_size)
        self.fc2 = torch.nn.Linear(self.fc1_size + self.num_actions, self.fc2_size)
        self.fc3 = torch.nn.Linear(self.fc2_size, 1)

        self.reset_parameters()

    def _init_hidden(self, layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)

        return -lim, lim

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*self._init_hidden(self.fc1))
        self.fc2.weight.data.uniform_(*self._init_hidden(self.fc2))
        self.fc3.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state, action):
        xs = F.relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))

        return self.fc3(x)


class Noise(object):
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(42)
        self.state = None
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx

        return self.state


class Agent(object):
    def __init__(self, params, num_states, num_actions, memory):
        self.params = params
        self.num_states = num_states
        self.num_actions = num_actions
        self.memory = memory

        # actor NN
        self.actor_local = ActorNetwork(self.params, self.num_states, self.num_actions).to(device)
        self.actor_target = ActorNetwork(self.params, self.num_states, self.num_actions).to(device)
        self.actor_optimizer = torch.optim.Adam(params=self.actor_local.parameters(), lr=self.params.learning_rate_actor)

        # actor NN
        self.critic_local = CriticNetwork(self.params, self.num_states, self.num_actions).to(device)
        self.critic_target = CriticNetwork(self.params, self.num_states, self.num_actions).to(device)
        self.critic_optimizer = torch.optim.Adam(params=self.critic_local.parameters(),
                                                 lr=self.params.learning_rate_critic,
                                                 weight_decay=self.params.weight_decay)

        #self.loss_function = torch.nn.MSELoss()

        self.noise = Noise(self.num_actions)

        self.total_steps = 0
        self.target_updated_count = 0

    def save_model(self, path):
        torch.save(self.nn.state_dict(), path)

    def load_model(self, path):
        self.nn.load_state_dict(torch.load(path))

    def reset(self):
        self.noise.reset()

    def get_action(self, state):
        self.total_steps += 1

        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()

        self.actor_local.train()

        action += self.params.epsilon * self.noise.sample()

        return np.clip(action, -1, 1)

    def step(self, state, action, new_state, reward, done, time_step):
        self.memory.push(state, action, new_state, reward, done)

        if time_step % 20 == 0:
            for i in range(10):
                self.optimize()

    def optimize(self):
        if len(self.memory) < self.params.batch_size:
            return

        states, actions, next_states, rewards, dones = self.memory.sample()

        states = Tensor(states).to(device)
        next_states = Tensor(next_states).to(device)

        rewards = Tensor(rewards).to(device)
        actions = Tensor(actions).to(device)
        dones = Tensor(dones).to(device)

        # critic update
        next_actions = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, next_actions)

        Q_targets = rewards + (1 - dones) * self.params.gamma * Q_targets_next

        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        #actor update
        action_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, action_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic_local, self.critic_target, self.params.tau)
        self.soft_update(self.actor_local, self.actor_target, self.params.tau)

        self.noise.reset()
        self.params.epsilon -= self.params.epsilon_decay

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
