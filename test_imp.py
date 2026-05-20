import numpy as np
from hw2 import calc_gini, calc_entropy

# Mock data
X = np.array([
    [1, 'a'],
    [2, 'a'],
    [3, 'b'],
    [4, 'b']
])
print("gini:", calc_gini(X))
print("entropy:", calc_entropy(X))
