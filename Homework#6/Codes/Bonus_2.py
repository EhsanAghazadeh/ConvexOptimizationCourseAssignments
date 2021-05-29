import cvxpy as cp
import numpy as np
from ellip_anomaly_data import *

vols = []
removed_points = []

n = len(X)
A = cp.Variable((n, n))
b = cp.Variable((n, 1))
ones = np.ones(len(X[0])).reshape(-1, 1)

# print(ones.shape)
# print(b.shape)
# print(A.shape)
# print(X.shape)
# print((A @ X).shape)
# print(b.T.shape)
# print(ones @ b.T)
# print(((A @ X).T + ones @ b.T).shape)
# print(cp.norm((A @ X).T + ones @ b.T).value)

constraints = [
               np.array([cp.norm(x) for x in (A @ X).T + ones @ b.T]) <= np.ones(len(X[0]))
]

# constraints = []
# for i in range(len(X)):
#     constraints += [
#                     cp.norm(((A @ X).T + ones @ b.T)[i]) <= 1
#     ]

obj = cp.Maximize(
    cp.log_det(A)
)

# for i in range(6):
prob = cp.Problem(obj, constraints)
prob.solve(solver=cp.SCS)

ones = np.ones(len(X[0])).reshape(-1, 1)
# [cp.norm(((A @ X).T + ones @ b.T)[i]) <= 1][0].dual_value

# prob.status

print(1 / np.linalg.det(A.value))

# 1 / np.linalg.det(A.value)