# %% Import Packages
import Taxi_Functions as tf
import gymnasium as gym
from tqdm import tqdm

# %% Defining Hyperparameters and some constants for Training
learning_rate = 0.05
n_episodes = 10_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2) 
final_epsilon = 0.1

N_DATA_POINTS = 25

# %%  Training
env = gym.make("Taxi-v3")
env.metadata['render_fps'] = 120
agent = tf.Taxi(
    env,    
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

### Iterate over the episodes and trains the agent using functions from Taxi_Functions.py

for episode in tqdm(range(n_episodes + 1)):
    obs, _ = env.reset()
    done = False
    score = 0

    while not done:
        action = agent.get_action(env, obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        score += reward

        agent.update(obs, action, reward, terminated, next_obs)

        

        done = terminated or truncated
        obs = next_obs
    
    if(episode % (n_episodes / N_DATA_POINTS) == 0):
        agent.save(f'Taxi\Runs\Generation_{episode}')
    agent.decay_epsilon()



