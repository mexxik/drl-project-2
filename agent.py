import math
import copy
import random
import torch
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Tensor = torch.Tensor


class Parameters(object):
    def __init__(self):
        self.num_episodes = 2000
        self.solve_score = 30

        self.replay_capacity = 100000
        self.batch_size = 64

        self.gamma = 0.99

        self.learning_rate_actor = 0.0001
        self.learning_rate_critic = 0.001

        self.weight_decay = 0

        self.tau = 0.001


class ExperienceReplay(object):
    def __init__(self, params):
        self.params = params

        self.memory = []
        self.position = 0

    def push(self, state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)

        if self.position >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self.position = (self.position + 1) % self.params.replay_capacity

    def sample(self):
        return zip(*random.sample(self.memory, self.params.batch_size))

    def __len__(self):
        return len(self.memory)


class NeuralNetwork(torch.nn.Module):
    def __init__(self, params, is_critic, num_states, num_actions):
        super(NeuralNetwork, self).__init__()

        self.params = params
        self.is_critic = is_critic
        self.num_states = num_states
        self.num_actions = num_actions

        self.fc1_size = 400
        self.fc2_size = 300

        self.fc1 = torch.nn.Linear(num_states, self.fc1_size)

        if self.is_critic:
            self.fc2 = torch.nn.Linear(self.fc1_size + self.num_actions, self.fc2_size)
            self.fc3 = torch.nn.Linear(self.fc2_size, 1)
        else:
            self.fc2 = torch.nn.Linear(self.fc1_size, self.fc2_size)
            self.fc3 = torch.nn.Linear(self.fc2_size, self.num_actions)

        self.reset_parameters()

    def _init_hidden(self, layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)

        return -lim, lim

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*self._init_hidden(self.fc1))
        self.fc1.weight.data.uniform_(*self._init_hidden(self.fc2))
        self.fc1.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state, action=None):
        if self.is_critic:
            xs = F.relu(self.fc1(state))
            x = torch.cat((xs, action), dim=1)
            x = F.relu(self.fc2(x))

            return self.fc3(x)
        else:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))

            return F.tanh(self.fc3(x))


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
        self.actor_local = NeuralNetwork(self.params, False, self.num_states, self.num_actions).to(device)
        self.actor_target = NeuralNetwork(self.params, False, self.num_states, self.num_actions).to(device)
        self.actor_optimizer = torch.optim.Adam(params=self.actor_local.parameters(), lr=self.params.learning_rate_actor)

        # actor NN
        self.critic_local = NeuralNetwork(self.params, True, self.num_states, self.num_actions).to(device)
        self.critic_target = NeuralNetwork(self.params, True, self.num_states, self.num_actions).to(device)
        self.critic_optimizer = torch.optim.Adam(params=self.critic_local.parameters(), lr=self.params.learning_rate_critic)

        #self.loss_function = torch.nn.MSELoss()

        self.noise = Noise(self.num_actions)

        self.total_steps = 0
        self.target_updated_count = 0

    def save_model(self, path):
        torch.save(self.nn.state_dict(), path)

    def load_model(self, path):
        self.nn.load_state_dict(torch.load(path))

    def get_epsilon(self, total_steps):
        epsilon = self.params.e_greedy_min + (self.params.e_greedy - self.params.e_greedy_min) * \
                  math.exp(-1. * total_steps / self.params.e_greedy_decay)
        return epsilon

    def get_action(self, state):
        self.total_steps += 1

        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()

        self.actor_local.train()

        action += self.noise.sample()

        return np.clip(action, -1, 1)

    def optimize(self):
        if len(self.memory) < self.params.batch_size:
            return

        state, action, next_state, reward, done = self.memory.sample()

        state = Tensor(state).to(device)
        next_state = Tensor(next_state).to(device)

        reward = Tensor(reward).to(device)
        action = Tensor(action).to(device)
        done = Tensor(done).to(device)

        # critic update
        next_action = self.actor_target(next_state)
        Q_target_next = self.critic_target(next_state, next_action)

        Q_target = reward + (1 - done) * self.params.gamma * Q_target_next

        Q_expected = self.critic_local(state, action)
        critic_loss = F.mse_loss(Q_expected, Q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #actor update
        action_pred = self.actor_local(state)
        actor_loss = -self.critic_local(state, action_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic_local, self.critic_target, self.params.tau)
        self.soft_update(self.actor_local, self.actor_target, self.params.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
