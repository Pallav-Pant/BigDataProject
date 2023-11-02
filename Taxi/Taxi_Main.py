
# %%
"""
Taxi with Reinforcement Learning
================================

"""
# Authors: Pallav Pant, Josh Lahr, Shannon Flaherty

# %%
# Imports and Environments
import Taxi_Functions as tfunc
import gymnasium as gym

# %%
# Taxi_Run_Test
# Checks if the taxi environment runs correctly.

#Render Mode = "human" allows us to see the taxi game being played.
# render_mode = None
# test_env = gym.make("Taxi-v3", render_mode = render_mode)


# episodes = 1000
# outcomes = []
# for episode in range(episodes):
#     state = test_env.reset()
#     terminated, truncated = False, False
#     score = 0
#     while not (terminated or score < -300):
#         action = test_env.action_space.sample()
#         obs, reward, terminated, truncated, info = test_env.step(action)
#         score += reward

#         print(f"Epsiode: {episode}, Score: {score}, Observation:{obs}")
#     outcomes.append(score)
#     print("End of Episode\n")

# test_env.close()
# outcomes.sort()
# outcomes.reverse()
# print(outcomes)


# %%
# Actual Program Start
if __name__ == '__main__':
    print("Start")
