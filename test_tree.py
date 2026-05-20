import numpy as np
import pandas as pd
from hw2 import DecisionTree, calc_entropy, calc_gini, chi_table, DecisionNode

# Check globals
data = pd.read_csv('agaricus-lepiota.csv')
data = data.dropna(axis=1)
X, y = data.drop('class', axis=1), data['class']
X = np.column_stack([X, y])

print("Data loaded. Shape:", X.shape)
