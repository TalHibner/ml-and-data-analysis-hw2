# Machine Learning & Data Analysis - Homework 2

## Overview
This repository contains Homework 2 for a machine learning and data analysis course. The assignment uses the mushroom dataset to practice classification and decision tree algorithms, with an emphasis on interpreting categorical data and understanding how model choices affect performance.

## Assignment Description
The task is to build and evaluate classifiers that distinguish between edible and poisonous mushrooms. Students will implement core parts of a decision tree learner, inspect feature importance, and compare results with baseline classifiers.

## Subjects Covered
- Supervised learning and binary classification
- Decision tree fundamentals and tree construction
- Information gain, entropy, and Gini impurity
- Categorical data handling and encoding strategies
- Model evaluation, accuracy, and overfitting
- Exploratory data analysis and visualization in a notebook format

## Key Challenges
- Working with an all-categorical dataset and converting mushroom attributes into a suitable format for learning algorithms
- Implementing decision tree splitting logic and stopping criteria correctly
- Comparing different split quality measures and understanding how they impact tree structure
- Avoiding overfitting on small or noisy data subsets
- Validating code with unit tests while tracking accuracy in the notebook

## Contents
- `agaricus-lepiota.csv` — mushroom dataset for training and evaluation
- `hw2.py` — main assignment code
- `hw2.ipynb` — notebook version with walkthrough, analysis, and visualization
- `scratch2.py` — scratch work and experiments
- `test_imp.py` — implementation tests
- `test_tree.py` — decision tree tests

## How to use
1. Open `hw2.ipynb` in Jupyter Notebook or JupyterLab.
2. Review the assignment instructions and complete the required functions in `hw2.py`.
3. Run the notebook cells to load data, train classifiers, and evaluate results.
4. Use the provided tests to verify your implementation.

## Requirements
- Python 3.x
- NumPy
- Pandas
- scikit-learn (if used by the notebook)

## Notes
- The notebook contains detailed steps and explanations.
- `hw2.py` should contain the final submission code.
- Use the tests to ensure your solution works before submitting.
