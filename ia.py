import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Initializing an empty table

def init_q_table(x: int, y: int) -> np.ndarray:
    """
    This function must return a 2D matrix containing only zeros for values.
    """
    return [[0 for _ in range(x)] for _ in range(y)]

qTable = init_q_table(5, 4)

print("Q-Table:\n" + str(qTable))

assert(np.mean(qTable) == 0)