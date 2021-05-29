# -*- coding: utf-8 -*-
"""HW#5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14x3Fb9ZZNHkmkak79Euoh_cdmeZiGcTj
"""

from nonlin_meas_data import *
import cvxpy as cp
import numpy as np
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt

"""##Q#2"""

row = np.zeros(m)
row[0] = -1
row[1] = 1
col = np.zeros(m-1)
col[0] = -1

B = toeplitz(col, row)

x = cp.Variable(n)
z = cp.Variable(m)

obj = cp.Minimize(cp.norm2(z - A @ x))
constraints = [
               (1 / beta) * (B @ y) <= B @ z,
               (1 / alpha) * (B @ y) >= B @ z
]

prob = cp.Problem(obj, constraints)
prob.solve()

print(f"x is: {x.value}")

plt.rcParams["figure.figsize"] = (10, 5)
plt.plot(z.value, y)
plt.xlabel(
    "y",
    fontdict= {
        'color': 'red',
        'size': 15
    }
)
plt.ylabel(
    "z_ml",
    fontdict= {
        'color': 'red',
        'size': 15
    }
)
plt.title(
    "Estimated function for phi",
    fontdict= {
        'color': 'red',
        'size': 20
    }
)
plt.show()

