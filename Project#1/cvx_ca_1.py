# -*- coding: utf-8 -*-
"""CVX_CA#1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sfMjrwkkuCSr0CXHEJNXd31kWvMTmnIB
"""

from robust_linear_models_data import *

import cvxpy as cp

def loss_sq(X, y, theta):
    return 0.5 * cp.pnorm(X @ theta - y, p=2)**2

def loss_abs(X, y, theta):
    return cp.pnorm(X @ theta - y, p=1)

def loss_norm(X, y, theta):
    coefs = np.diag(np.array([1. / max(np.linalg.norm(x, ord=2), 1) for x in X]))
    return cp.pnorm(
        coefs @ (X @ theta - y),
        p=1
    )

"""#(a)

###X_std, y_std
"""

theta = cp.Variable(n)
problem = cp.Problem(cp.Minimize(loss_sq(X_std, y_std, theta)))
problem.solve()

print("Part a Results:")
print("Case: X_std, y_std")
print("norm-2 of theta-theta_gen = {}".format(np.linalg.norm(theta.value - theta_gen, ord=2)))

"""###X_std, y_outliers"""

theta = cp.Variable(n)
problem = cp.Problem(cp.Minimize(loss_sq(X_std, y_outliers, theta)))
problem.solve()

print("Case: X_std, y_outliers")
print("norm-2 of theta-theta_gen = {}".format(np.linalg.norm(theta.value - theta_gen, ord=2)))

"""###X_outliers, y_outliers"""

theta = cp.Variable(n)
problem = cp.Problem(cp.Minimize(loss_sq(X_outliers, y_outliers, theta)))
problem.solve()

print("Case: X_outliers, y_outliers")
print("norm-2 of theta-theta_gen = {}".format(np.linalg.norm(theta.value - theta_gen, ord=2)))

"""###X_outliers, y_std"""

theta = cp.Variable(n)
problem = cp.Problem(cp.Minimize(loss_sq(X_outliers, y_std, theta)))
problem.solve()

print("Case: X_outliers, y_std")
print("norm-2 of theta-theta_gen = {}".format(np.linalg.norm(theta.value - theta_gen, ord=2)))


print("Part b results:")
"""#(b)

###X_std, y_std
"""

theta = cp.Variable(n)
problem = cp.Problem(cp.Minimize(loss_abs(X_std, y_std, theta)))
problem.solve()

print("Case: X_std, y_std")
print("norm-2 of theta-theta_gen = {}".format(np.linalg.norm(theta.value - theta_gen, ord=2)))

"""###X_std, y_outliers"""

theta = cp.Variable(n)
problem = cp.Problem(cp.Minimize(loss_abs(X_std, y_outliers, theta)))
problem.solve()

print("Case: X_std, y_outliers")
print("norm-2 of theta-theta_gen = {}".format(np.linalg.norm(theta.value - theta_gen, ord=2)))

"""###X_outliers, y_outliers"""

theta = cp.Variable(n)
problem = cp.Problem(cp.Minimize(loss_abs(X_outliers, y_outliers, theta)))
problem.solve()

print("Case: X_outliers, y_outliers")
print("norm-2 of theta-theta_gen = {}".format(np.linalg.norm(theta.value - theta_gen, ord=2)))

"""###X_outliers, y_std"""

theta = cp.Variable(n)
problem = cp.Problem(cp.Minimize(loss_abs(X_outliers, y_std, theta)))
problem.solve()

print("Case: X_outliers, y_std")
print("norm-2 of theta-theta_gen = {}".format(np.linalg.norm(theta.value - theta_gen, ord=2)))


print("Part c results:")
"""#(c)

###X_std, y_std
"""

theta = cp.Variable(n)
problem = cp.Problem(cp.Minimize(loss_norm(X_std, y_std, theta)))
problem.solve()

print("Case: X_std, y_std")
print("norm-2 of theta-theta_gen = {}".format(np.linalg.norm(theta.value - theta_gen, ord=2)))

"""###X_std, y_outliers"""

theta = cp.Variable(n)
problem = cp.Problem(cp.Minimize(loss_norm(X_std, y_outliers, theta)))
problem.solve()

print("Case: X_std, y_outliers")
print("norm-2 of theta-theta_gen = {}".format(np.linalg.norm(theta.value - theta_gen, ord=2)))


theta = cp.Variable(n)
problem = cp.Problem(cp.Minimize(loss_norm(X_outliers, y_outliers, theta)))
problem.solve()

print("Case: X_outliers, y_outliers")
print("norm-2 of theta-theta_gen = {}".format(np.linalg.norm(theta.value - theta_gen, ord=2)))

theta = cp.Variable(n)
problem = cp.Problem(cp.Minimize(loss_norm(X_outliers, y_std, theta)))
problem.solve()

print("Case: X_outliers, y_std")
print("norm-2 of theta-theta_gen = {}".format(np.linalg.norm(theta.value - theta_gen, ord=2)))

