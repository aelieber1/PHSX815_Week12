"""
PHSX 815 Homework 12

author: @aelieber1
date: May 1, 2023

Purpose: 
Implement either nearest neighbor or Gaussian Kernel density estimation. This should include include:
    - Simulate a dataset sampled from some function in a real-valued variable (you can use any previous HW or project, 1D is fine)
    - Estimate the density of this dataset using either method.
    - Draw your estimate and the dataset together. What values of the density estimator parameters seem to give the "best" results?

Sources:
    - https://stackabuse.com/kernel-density-estimation-in-python-using-scikit-learn/
    - 
"""

# Import necessary external packages
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from scipy.stats import norm

# Sample data from a gaussian
def make_data(N, f=0.3, rseed=1):
    rand = np.random.RandomState(rseed)
    x = rand.randn(N)
    x[int(f * N):] += 5
    return x

# sample 500 data points
x = make_data(500)

# Kernel Density Estimation
x_d = np.linspace(-4, 12, 500)
density = sum(norm(xi).pdf(x_d) for xi in x)

# Plot density estimates
plt.fill_between(x_d, density, alpha=0.5, label="density estimate")
plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)
plt.axis([-4, 12, -0.2, 100])

# Plot histogram of data
plt.hist(x, bins=30, label='randomly sampled gaussian data')

plt.legend()
plt.xlabel("{x} data")
plt.title("Kernel density estimate and dataset comparison")
plt.ylabel(" ")
plt.show()

