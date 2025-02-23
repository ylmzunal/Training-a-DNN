"""
Yilmaz Ünal Student ID: 2023719108
This script collects environment data for training a deep learnin model and saves it to a file.
"""

import os
import numpy as np
import pickle
from homework1 import Hw1Env

# Create the environment
env = Hw1Env(render_mode="gui")

# Collect data (Run only once)
data = []
i= 0
for _ in range(10):
    print(f"{i}. iteration")
    i +=1
    env.reset()
    action_id = np.random.randint(4)
    _, img_before = env.state()
    env.step(action_id)
    pos_after, img_after = env.state()
    env.reset()
    data.append((img_before, action_id, pos_after, img_after))


# # Define the dataset path
# dataset_path = "dataset.pkl"

# # Save the collected data to a file
# with open(dataset_path, "wb") as f:
#     pickle.dump(data, f)

# print(f"Dataset saved to {dataset_path}")
