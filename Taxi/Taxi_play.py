import Taxi_Functions as tf
import gymnasium as gym
from collections import defaultdict


N_RUNS = 15

env = gym.make('Taxi-v3', render_mode = 'human')

agent = tf.Taxi(env=env)

agent.load('Taxi\Runs\Generation_10000')

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