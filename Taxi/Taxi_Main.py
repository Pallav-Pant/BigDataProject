# %% Import Packages
import Taxi_Functions as tf
import gymnasium as gym
from tqdm import tqdm
from collections import defaultdict

# %% Defining Hyperparameters for Training
learning_rate = 0.001
n_episodes = 1000000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2) 
final_epsilon = 0.1

# %% Main
if __name__ == '__main__':
    env = gym.make("Taxi-v3")
    agent = tf.Taxi(
        env,    
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )
    
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
    for episode in tqdm(range(n_episodes)):
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
        
        agent.decay_epsilon()

    agent.save('Final_Shit')

    env.close()
## ====================================================================================== ##




    test_env = gym.make("Taxi-v3", render_mode = None)
    test_count = 1000
    outcomes = defaultdict()

    for x in range(test_count):
        score = 0
        obs, _ = test_env.reset()
        done = False
        while not done:
            action = agent.act(test_env, obs)
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





