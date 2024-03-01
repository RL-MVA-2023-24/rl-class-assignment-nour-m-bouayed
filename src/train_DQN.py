from gymnasium.wrappers import TimeLimit
import numpy as np
import random
import torch
from torch import nn
import os
from copy import deepcopy
from env_hiv import HIVPatient
from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()

class DQN(torch.nn.Module):
    def __init__(self, input_dim=env.observation_space.shape[0], hidden_dim=512, output_dim=env.action_space.n, depth = 5):
        super(DQN, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 1)])
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)
    
class ProjectAgent:
    def __init__(self, save_name='DQN', train=False, config={}):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model =  DQN().to(device)
        self.path = os.path.join(os.getcwd(),f'models/{save_name}.pt')
        self.target_model = deepcopy(self.model).to(device)
        if train:
            self.nb_actions = config['nb_actions']
            self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
            self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
            buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
            self.memory = ReplayBuffer(buffer_size,device)
            self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
            self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
            self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
            self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
            self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
            self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
            lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
            self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
            self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
            self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
            self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
            self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
      
    def act(self, observation, use_random=False):
        return greedy_action(self.model, observation)

    def save(self):
        torch.save(self.model.state_dict(), self.path)
        return 

    def load(self):
        device = torch.device('cpu')
        self.model = DQN()
        self.model.load_state_dict(torch.load(self.path, map_location=device))
        self.model.eval()
        return 
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, env, max_episode, evaluate_every=10):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                if episode % evaluate_every == 0:
                    self.evaluate()
                    self.save()
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:e}'.format(episode_cum_reward),
                      sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return
    
    def evaluate(self):
        score_agent = evaluate_HIV(agent=self, nb_episode=1)
        print(f'score_agent = {score_agent:e}')

if __name__=="__main__":    
    # DQN config
    config = {'nb_actions': env.action_space.n,
            'learning_rate': 0.001,
            'gamma': 0.95,
            'buffer_size': 1000000,
            'epsilon_min': 0.01,
            'epsilon_max': 1.,
            'epsilon_decay_period': 1000,
            'epsilon_delay_decay': 20,
            'batch_size': 20,
            'gradient_steps': 1,
            'update_target_strategy': 'replace', # or 'ema'
            'update_target_freq': 50,
            'update_target_tau': 0.005,
            'criterion': torch.nn.SmoothL1Loss(),
            'monitoring_nb_trials': 50}
    
    max_episode = 150
    # Train agent
    agent = ProjectAgent(train=True, config=config)
    agent.train(env, max_episode)

