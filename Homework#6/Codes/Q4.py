import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

m, n = 3, 3
k = 201
t = -3 + 6 * np.arange(k) / (k - 1)
y = np.exp(t)
Tpowers = np.concatenate([np.ones((k, 1)), t.reshape(-1, 1), (t**2).reshape(-1, 1)], axis=1)
a = cp.Variable(m)
b = cp.Variable(n-1)
alpha = cp.Parameter(nonneg=True)
lhs = cp.abs(Tpowers * a - np.diag(y) * (Tpowers * cp.hstack([1, b])))
rhs = alpha * (np.diag(y) * (Tpowers * cp.hstack([1, b])))
problem = cp.Problem(cp.Minimize(0), [lhs <= rhs])

l, u = 0, np.exp(3) # initial upper and lower bounds
bisection_tol = 1e-3 # bisection tolerance
while u - l >= bisection_tol:
    alpha.value = (l + u) / 2
    # solve the feasibility problem
    problem.solve()
    if problem.status == "optimal":
        u = alpha.value
        a_opt = a.value
        b_opt = b.value
        objval_opt = alpha.value
    else:
        l = alpha.value

y_fit = ((Tpowers @ a_opt) / (Tpowers @ np.concatenate([[1], b_opt])))
plt.figure()
plt.plot(t, y, 'b', t, y_fit, 'r+')
plt.xlabel('t')
plt.ylabel('y')
plt.show()
plt.figure()
plt.plot(t, y_fit - y)
plt.xlabel('t')
plt.ylabel('err')
plt.show()