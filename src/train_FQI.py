from gymnasium.wrappers import TimeLimit
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import random
import os
import torch
import pickle

from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self, model_name):
        self.env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)
        self.path = os.path.join(os.getcwd(),f'models/{model_name}.pkl')

    def act(self, observation, use_random=False):
        return self.greedy_policy(observation)

    def save(self):
        with open(self.path,'wb') as f:
            pickle.dump(self.Q,f)

    def load(self):
        with open(self.path, 'rb') as f:
            self.Q = pickle.load(f)

    def collect_samples(self, nb_samples, policy='random', disable_tqdm=False):
        s, _ = self.env.reset()
        if nb_samples> 200:
            repetitions = nb_samples//200
            horizon = 200
        else :
            repetitions = 1
            horizon = nb_samples
        S = []
        A = []
        R = []
        S2 = []
        with tqdm(total=horizon*repetitions, position=0, leave=True) as pbar:
            for _ in range(repetitions):
                for _ in range(horizon):
                    pbar.update(1)
                    if policy=='random':
                        a = self.env.action_space.sample()
                    else:
                        a = self.greedy_policy(s)
                    s2, r, _, _, _ = self.env.step(a)
                    #dataset.append((s,a,r,s2,done,trunc))
                    S.append(s)
                    A.append(a)
                    R.append(r)
                    S2.append(s2)
                    s = s2
        S = np.array(S)
        A = np.array(A).reshape((-1,1))
        R = np.array(R)
        S2= np.array(S2)
        return S, A, R, S2

    def greedy_policy(self,s, store_seen=False):
        Qsa = []
        for a in range(self.env.action_space.n):
            sa = np.append(s,a).reshape(1, -1)
            Qsa.append(self.Q.predict(sa))
        return np.argmax(Qsa)

    def train(self, nb_samples, epochs, iterations_per_epoch, gamma, disable_tqdm=False):
        for epoch in range(epochs):
            print(f'********************** EPOCH {epoch +1} **********************')
            print('1. Collecting samples')
            if epoch == 0:
                S, A, R, S2 = self.collect_samples(nb_samples, policy='random')
            else:
                S_, A_, R_, S2_ = self.collect_samples(200, policy='learned')
                S, A, R, S2 = np.append(S,S_,axis=0)[200:], np.append(A,A_,axis=0)[200:], np.append(R,R_,axis=0)[200:], np.append(S2,S2_,axis=0)[200:]
            nb_samples = S.shape[0]
            nb_actions = env.action_space.n
            SA = np.append(S,A,axis=1)
            print('2. Learning')
            for iter in tqdm(range(iterations_per_epoch), disable=disable_tqdm, position=0, leave=True):
                if epoch==0 and iter==0:
                    value=R.copy()
                else:
                    Q2 = np.zeros((nb_samples,nb_actions))
                    for a2 in range(nb_actions):
                        A2 = a2*np.ones((S.shape[0],1))
                        S2A2 = np.append(S2,A2,axis=1)
                        Q2[:,a2] = self.Q.predict(S2A2)
                    max_Q2 = np.max(Q2,axis=1)
                    value = R + gamma*max_Q2
                self.Q = RandomForestRegressor()
                self.Q.fit(SA,value)
            print('3. Evaluating')
            self.evaluate()
        self.save()

    def evaluate(self):
        score_agent = evaluate_HIV(agent=self, nb_episode=1)
        print(f'score_agent = {score_agent:e}')
        score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=1)
        print(f'score_agent_dr = {score_agent_dr:e}')

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    seed_everything(seed=42)
    # Initialization of the agent. Replace DummyAgent with your custom agent implementation.
    agent = ProjectAgent('FQI')
    agent.train(10000, 1, 1000, 0.9)
#     agent.train(10,1,1,0.9)
    agent.load()
    # Keep the following lines to evaluate your agent unchanged.
    score_agent: float = evaluate_HIV(agent=agent, nb_episode=1)
    score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=1)
    with open(file="score.txt", mode="w") as f:
        f.write(f"{score_agent}\n{score_agent_dr}")