# %% Import Packages
import Taxi_Functions as tf
import gymnasium as gym
from collections import defaultdict

# %% Testing
G_NAME = 'Taxi\Generation_'
N_RUNS = 10000
env = gym.make('Taxi-v3')

generations = [0,1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 'Final']

generation_data = defaultdict()

for x in generations:
    agent = tf.Taxi(env=env)
    agent_name = G_NAME+str(x)
    if (str(x) == 'Final'):
        agent_name = 'Taxi\Final'
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
for x in generations:
    print(f'Generation: {x}\t Average Score: {generation_data[x]}\n')