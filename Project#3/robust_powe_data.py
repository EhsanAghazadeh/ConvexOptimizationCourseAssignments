# robust_power_data.py
#
# Data file for the robust power problem

import numpy as np

m = 20
n = 30

L = np.zeros((m, n))  # L[i, j] is length between station i and destination node j

# Lengths between nodes related to distance between i, j, but with some
# modification to keep solutions a bit more unique.
for i in range(1, m+1):
    for j in range(1, n+1):
        a = i - .5;
        b = (j - 1.0) / 2
        mult = 2 + (i - 1.0) / m
        dist = np.sqrt(a - b) if a >= b else np.sqrt(mult * (b - a))
        L[i-1, j-1] = 1 + dist

# Transmission loss rate
alpha = .15

# Power capacities on each power station
c = 5 * np.ones(m)

# Usage level at node
u = 1.0 + np.linspace(0, 1, num=n)

print(L)
print(u)