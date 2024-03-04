from gymnasium.wrappers import TimeLimit
import numpy as np
from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population
import random
import torch
from torch import nn
import os
from copy import deepcopy

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def act(self, observation, use_random=False):
        return np.random.randint(4)

    def save(self, path):
        pass

    def load(self):
        pass

env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()

class DQN(torch.nn.Module):
    def __init__(self, input_dim=env.observation_space.shape[0], hidden_dim=256, output_dim=env.action_space.n, depth = 6):
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
    def __init__(self, save_name='model-one-2.1e+10--dr-2.2e+10', train=False, config={}):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model =  DQN().to(self.device)
        self.load_path = os.path.join(os.getcwd(),f'models/{save_name}.pt')
        self.save_path =self.load_path
        self.target_model = deepcopy(self.model).to(self.device)
        if train:
            self.nb_actions = config['nb_actions']
            self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
            self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
            self.buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
            self.memory = ReplayBuffer(self.buffer_size,self.device)
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
        torch.save(self.model.state_dict(), self.save_path)

    def load(self, train=False):
        self.model = DQN().to(self.device)
        self.model.load_state_dict(torch.load(self.load_path, map_location=self.device))
        if train:
            lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
            self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
            self.model.train()
        else:
            self.model.eval()
    
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
    
    def train(self, max_episode, evaluate_every=10):
        self.model.train()
        env = TimeLimit(env=HIVPatient(domain_randomization=True),max_episode_steps=200)
        switched = False
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        previous_best_score, previous_best_score_dr = 0, 0
        random = "1"
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
                #LOGGING----------------------------
                if episode > 120:
                    score, score_dr = self.evaluate(dr=True)
                    if score > previous_best_score:
                        torch.save(self.model.state_dict(), f"model-one-{score:.1e}--dr-{score_dr:.1e}.pt")
                        previous_best_score = score
                    if score_dr >previous_best_score_dr:
                        torch.save(self.model.state_dict(), f"model-one-{score:.1e}--dr-{score_dr:.1e}.pt")
                        previous_best_score_dr = score_dr
                if np.random.rand() < 0.4:
                    random = "0"
                    env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)
                else :
                    random = "1"
                    env = TimeLimit(env=HIVPatient(domain_randomization=True), max_episode_steps=200)
                episode += 1
                print("dr = ", random, 
                      "-- Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:.3e}'.format(episode_cum_reward),
                      sep='')
                state, _ = env.reset()
                episode_cum_reward = 0
                self.save()
            else:
                state = next_state
    
    def evaluate(self, dr=False):
        score_agent = evaluate_HIV(agent=self, nb_episode=1)
        print(f'score_agent = {score_agent:.3e}')
        if dr:
            score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=15)
            print(f'score_agent_dr = {score_agent_dr:.3e}')
        else:
            score_agent_dr = None
        return score_agent, score_agent_dr

if __name__=="__main__":   
    
    seed_everything(seed=42)
    # DQN config
    config = {'nb_actions': env.action_space.n,
            'learning_rate': 0.0005,
            'gamma': 0.99,
            'buffer_size': 1000000,
            'epsilon_min': 0.01,
            'epsilon_max': 1.0,
            'epsilon_decay_period': 20000,
            'epsilon_delay_decay': 100,
            'batch_size': 800,
            'gradient_steps': 3,
            'update_target_strategy': 'replace', # or 'ema'
            'update_target_freq': 400,
            'criterion': torch.nn.SmoothL1Loss()}
    
    max_episode = 300
    # Train agent
    agent = ProjectAgent(train=True, config=config)
#     agent.load(train=True)
    agent.train(max_episode)