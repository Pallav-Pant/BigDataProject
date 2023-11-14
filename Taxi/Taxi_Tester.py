# %% Import Packages
import Taxi_Functions as tf
import gymnasium as gym
from collections import defaultdict
import matplotlib.pyplot as plt

# %% Training Data
N_TRAIN = 10_000
N_SAVES = 25

# %% Testing
G_NAME = 'Taxi\Runs\Generation_'
N_RUNS = 10000 # No of runs for testing
env = gym.make('Taxi-v3')

n = 0
generations = []
while n < N_TRAIN+1:
    generations.append(int(n))
    n+= N_TRAIN / N_SAVES


generation_data = defaultdict()

for x in generations:
    agent = tf.Taxi(env=env)
    agent_name = G_NAME+str(x)
    
    agent.load(agent_name)
    outcomes = []
    for i in range(N_RUNS):
        

        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.get_best_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            score += reward

            done = terminated or truncated
            obs = next_obs
        outcomes.append(score)

    generation_data[x] = tf.average(outcomes)

env.close()
tf.save_data(generation_data, 'Taxi\Taxi_Run_Data')
for x in generations:
    print(f'Generation: {x}\t Average Score: {generation_data[x]}\n')


### Labels and plots

xlabel = 'Generation Number'
ylabel = 'Average Score'
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.plot(*zip(*sorted(generation_data.items())))
plt.savefig('Taxi/Taxi_Runs_Data.png')
plt.show()