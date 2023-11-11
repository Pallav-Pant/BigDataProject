# %% Import Packages
import Taxi_Functions as tf
import gymnasium as gym
from collections import defaultdict

# %% Hyperparameters Unncessary
learning_rate = 0.01
n_episodes = 10_000
start_epsilon = 0.1
epsilon_decay = start_epsilon / (n_episodes / 2) 
final_epsilon = 0.1


# %%



test_env = gym.make("Taxi-v3")

test_agent = tf.Taxi(
    test_env,    
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

test_agent.load('Final_Shit')
print(test_agent.q_values)

test_count = 10
outcomes = defaultdict()

for x in range(test_count):
    score = 0
    obs, _ = test_env.reset()
    done = False
    while not done:
        action = test_agent.get_action(test_env, obs)
        next_obs, reward, terminated, truncated, _ = test_env.step(action)
        score += reward

        done = terminated or truncated
        obs = next_obs
outcomes[x] = score
test_env.close()

avg_test_score = 0
for x in outcomes:
    #print(f"Test {x}: {outcomes[x]}")
    avg_test_score += outcomes[x]

avg_test_score/=test_count
print(f"Average test score: {avg_test_score}")


