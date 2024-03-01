from gymnasium.wrappers import TimeLimit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os
from evaluate import evaluate_HIV, evaluate_HIV_population

from tqdm import trange

from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class policyNetwork(nn.Module):
    def __init__(self, env,  hidden_dim=256, depth = 6):
        super().__init__()
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 1)])
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(dim=0)
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        action_scores = self.output_layer(x)
        return F.softmax(action_scores,dim=1)

    def sample_action(self, x):
        probabilities = self.forward(x)
        action_distribution = Categorical(probabilities)
        return action_distribution.sample().item()

    def log_prob(self, x, a):
        probabilities = self.forward(x)
        action_distribution = Categorical(probabilities)
        return action_distribution.log_prob(a)
    
# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

class ProjectAgent:
    def __init__(self, save_name='PG', config={}):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy_network = policyNetwork(env).to(device)
        self.path = os.path.join(os.getcwd(),f'models/{save_name}.pt')
        # self.device = "cuda" if next(policy_network.parameters()).is_cuda else "cpu"
        self.scalar_dtype = next(policy_network.parameters()).dtype
        self.policy = policy_network
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.99
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = torch.optim.Adam(list(self.policy.parameters()),lr=lr)
        self.nb_episodes = config['nb_episodes'] if 'nb_episodes' in config.keys() else 1
    
    def one_gradient_step(self, env):
        # run trajectories until done
        episodes_sum_of_rewards = []
        states = []
        actions = []
        returns = []
        for ep in range(self.nb_episodes):
            x,_ = env.reset()
            rewards = []
            episode_cum_reward = 0
            while(True):
                a = self.policy.sample_action(torch.as_tensor(x, dtype=torch.float))
                y,r,done,trunc,_ = env.step(a)
                states.append(x)
                actions.append(a)
                rewards.append(r)
                episode_cum_reward += r
                x=y
                if done: 
                    # The condition above should actually be "done or trunc" so that we 
                    # terminate the rollout also if trunc=True.
                    # But then, our return-to-go computation would be biased as we would 
                    # implicitly assume no rewards can be obtained after truncation, which 
                    # is wrong.
                    # We leave it as is for now (which means we will call .step() even 
                    # after trunc=True) and will discuss it later.
                    # Compute returns-to-go
                    new_returns = []
                    G_t = 0
                    for r in reversed(rewards):
                        G_t = r + self.gamma * G_t
                        new_returns.append(G_t)
                    new_returns = list(reversed(new_returns))
                    returns.extend(new_returns)
                    episodes_sum_of_rewards.append(episode_cum_reward)
                    break
        # make loss
        returns = torch.tensor(returns)
        log_prob = self.policy.log_prob(torch.as_tensor(np.array(states)),torch.as_tensor(np.array(actions)))
        loss = -(returns * log_prob).mean()
        # gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return np.mean(episodes_sum_of_rewards)

    def train(self, env, nb_rollouts):
        avg_sum_rewards = []
        for ep in range(nb_rollouts):
            ep_return = self.one_gradient_step(env)
            avg_sum_rewards.append(ep_return)
            print("Rollout ", '{:3d}'.format(ep),  
                ", episode return ", '{:e}'.format(ep_return),
                sep='')
            self.evaluate()
            self.save()
        return avg_sum_rewards
    
    def act(self, observation, use_random=False):
        return self.policy.sample_action(torch.as_tensor(observation))

    def save(self):
        torch.save(self.policy.state_dict(), self.path)
        return 

    def load(self):
        device = torch.device('cpu')
        self.policy = policyNetwork()
        self.policy.load_state_dict(torch.load(self.path, map_location=device))
        self.policy.eval()
        return 
    
    def evaluate(self):
        score_agent = evaluate_HIV(agent=self, nb_episode=1)
        print(f'score_agent = {score_agent:e}')
        score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=5)
        print(f'score_agent_dr = {score_agent_dr:e}')


#env = gym.make("LunarLander-v2", render_mode="rgb_array")
config = {'gamma': .99,
          'learning_rate': 0.01,
          'nb_episodes': 10
         }

agent = ProjectAgent(config)
returns = agent.train(env,250)
