import numpy as np
from rl.utils.general import argmax_ties_random

rng = np.random.default_rng()
column_vector = rng.uniform(0, 5, (5, 1))
print(column_vector)
q = argmax_ties_random(column_vector)
print(f"Best q is:{q}")