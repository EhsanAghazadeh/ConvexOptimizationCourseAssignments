import numpy as np
import cvxpy as cp
from sphere_fit_data import *
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


n = 2

# Define and solve the CVXPY problem.
A = np.array(
    [
     np.concatenate([2*val, [1]]) for val in U.T
    ]
)
b = np.array(
    [
     np.linalg.norm(val, ord=2)**2 for val in U.T
    ]
)

x = cp.Variable(n+1)
cost = cp.sum_squares(A @ x - b)
prob = cp.Problem(cp.Minimize(cost))
prob.solve()

x_c = x.value[:2]
r = (x.value[2] + np.linalg.norm(x_c, ord=2)**2)**0.5

print(f"center is: {x_c}")
print(f"radius is: {r}")


fig = plt.figure()
ax1 = fig.add_subplot()
ax1.add_artist(Circle(x_c, r))
fig.savefig('perfectCircle1.png')