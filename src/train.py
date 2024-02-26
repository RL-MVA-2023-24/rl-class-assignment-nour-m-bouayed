from gymnasium.wrappers import TimeLimit
import numpy as np
from env_hiv import HIVPatient

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

# def act(self, observation, use_random=False):

#     device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
#     with torch.no_grad():
#         Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
#         return torch.argmax(Q).item()

#     # return 0


# def save(self, path):
#     self.path = path + "/model2.pt"
#     torch.save(self.model.state_dict(), self.path)
#     return 

# def load(self):
#     device = torch.device('cpu')
#     self.path = os.getcwd() + "/model2.pt"
#     self.model = self.network({}, device)
#     self.model.load_state_dict(torch.load(self.path, map_location=device))
#     self.model.eval()
#     return 