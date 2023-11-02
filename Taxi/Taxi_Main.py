
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

##Render Mode = "human" allows us to see the taxi game being played.
render_mode = None
test_env = gym.make("Taxi-v3", render_mode = render_mode)

#Epsisodes: Number of times to run the simulation
episodes = 10000
#Bounding Score: The maximum negative score the player can achieve before being reset 
bound_score = -300
#Array to collect final scores of players
outcomes = []


for episode in range(episodes):
    state = test_env.reset()
    terminated, truncated = False, False
    score = 0
    while not (terminated or score < bound_score):
        action = test_env.action_space.sample()
        obs, reward, terminated, truncated, info = test_env.step(action)
        score += reward

    outcomes.append(score)
#    print("End of Episode\n")

test_env.close()
# Close the test environment

passper = 0
for x in outcomes:
    if(x > bound_score):
        passper+=1

print(f"{passper/episodes * 100}% chance of completing on random")
# about 1% with a socre bound of 300

# %%
# Actual Program Start
if __name__ == '__main__':
    print("Start")
