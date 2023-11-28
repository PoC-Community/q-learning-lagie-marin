import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


LEARNING_RATE = 0.05
DISCOUNT_RATE = 0.99

def q_function(q_table: np.ndarray, state: int, action: int, reward: int, newState: int) -> float:
    """
    This function must implement the q_function equation pictured above.

    It should return the updated q-table.
    """
    """q_new = q_table + LEARNING_RATE * (reward + DISCOUNT_RATE * )"""
    return q_table[state, action] + LEARNING_RATE * (reward + DISCOUNT_RATE * max(q_table[newState, :] - q_table[state, action]))

# Initializing an empty table

def init_q_table(x: int, y: int) -> np.ndarray:
    """
    This function must return a 2D matrix containing only zeros for values.
    """
    return np.zeros((x, y))

def random_action(env):
    return env.action_space.sample()

def game_loop(env: gym.Env, q_table: np.ndarray, state: int, action: int) -> tuple:
    return None

# write some code to load and make the FrozenLake environment:
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode="rgb_array")
total_actions = env.action_space.n

assert(total_actions == 4), f"There are a total of four possible actions in this environment. Your answer is {total_actions}"

observation, info = env.reset()

# Performing an action
action = random_action(env)
observation, reward, done, _, info = env.step(action)

# Displaying the first frame of the game
plt.imshow(env.render())

# Printing game info
print(f"actions: {env.action_space.n}\nstates: {env.observation_space.n}")
print(f"Current state: {observation}")

# Closing the environment
env.close()
q_table = init_q_table(5,4)
q_table[0, 1] = q_function(q_table, state=0, action=1, reward=-1, newState=3)


assert(q_table[0, 1] == -LEARNING_RATE), f"The Q function is incorrect: the value of qTable[0, 1] should be -{LEARNING_RATE}"
print("Q-Table after action:\n" + str(q_table))
