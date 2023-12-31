# %% Import Packages
import gymnasium as gym
from collections import defaultdict
import numpy as np
import json

# %% Defining the Taxi class
class Taxi:
    def __init__(
            self,
            env: gym.wrappers.time_limit.TimeLimit,
            learning_rate: float = 0.01,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 0,
            final_epsilon: float = 0.1,
            discount_factor: float = 0.95,
    ):
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

    ### Gets the either best action or random action to input for training

    def get_action(self,env, obs: int) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    ### Gets the best action to input for testing

    def get_best_action(self, obs: int) -> int:
        return int(np.argmax(self.q_values[obs]))
    
    ### Updates the data to better fit new information and creates better q-values

    def update(
        self,
        obs: int,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: int,
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

    ### Decays the epsilon value to ensure that the agent explores less as it learns

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    ### Saves the q-values to a json file

    def save(self, file_name):
        save_agent = defaultdict()
        for x in self.q_values:
            save_agent[x] = self.q_values[x].tolist()
        
        with open(file_name+'.json', 'w') as f:
            json.dump(save_agent, f)
        
    ### Loads the q-values from a json file

    def load(self, file_name):
        with open(file_name+'.json', 'r') as f:
            data = json.load(f)
            for x in data:
                self.q_values[int(x)] = np.array(data[x])
        
           
            
### Averages a list of numbers

def average(lst):
    return sum(lst)/len(lst)

### Saves data to a json file

def save_data(data, file_name):
    with open(file_name+'.json', 'w') as f:
        json.dump(data, f)

