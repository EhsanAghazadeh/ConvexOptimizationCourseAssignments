import numpy as np
from utils import *
import cvxpy as cp


def solve_lp_newton_method(
        A,
        b,
        c,
        x_0,
        alpha=0.01,
        beta=0.5,
        max_iter=100,
        epsilon=10e-3
):
    if np.min(x_0) <= 0 or np.linalg.norm(A @ x_0 - b) > epsilon:
        print("ERROR: x_0 is not feasible")
        return None

    x = x_0
    lambda_hist = []
    for iter_num in range(max_iter):
        print(f"iter number: {iter_num}")
        H = np.diag(x.reshape(-1, ) ** -2)
        g = c - x ** -1

        H_inv = np.diag(x.reshape(-1, ) ** 2)
        w = np.linalg.lstsq(A @ (H_inv @ A.T), -1 * A @ (H_inv @ g))[0]

        delta_x_nt = -1 * (H_inv @ (A.T @ w + g))

        lambda_square = -1 * (g.T @ delta_x_nt)[0][0]
        lambda_hist.append(lambda_square / 2)
        if lambda_square / 2 <= epsilon:
            break

        t = 1
        while np.min(x + t * delta_x_nt) <= 0:
            t = beta * t

        while (c.T @ (t * delta_x_nt) - np.sum(np.log(x + t * delta_x_nt)) + np.sum(np.log(x)))[0][0] > (alpha * t * (
                g.T @ delta_x_nt))[0][0]:
            t = beta * t

        x = x + t * delta_x_nt

        print(f"lambda history: {lambda_hist}")

    if iter_num == max_iter - 1:
        print("ERROR: the maximum number for iteration number reached.")
        lambda_hist = None
        x_optimal = None
        nu_optimal = None
    else:
        x_optimal = x
        nu_optimal = w

    return x_optimal, nu_optimal, lambda_hist


if __name__ == '__main__':
    m = 100
    n = 500
    np.random.seed(2)
    A = np.concatenate(
        [
            np.random.randn(m - 1, n),
            np.ones((1, n))
        ],
        axis=0
    )
    x_0 = np.random.rand(n, 1)
    b = A @ x_0
    c = np.random.rand(n, 1)

    x_optimal, nu_optimal, lambda_hist = solve_lp_newton_method(
        A,
        b,
        c,
        x_0,
        alpha=0.01,
        beta=0.5,
        max_iter=100,
        epsilon=1e-3
    )

    print(f"x optimal: {x_optimal}")
    print(f"lambda history: {lambda_hist}")

    x = cp.Variable((n, 1))
    constraints = [
        A * x == b
    ]
    obj = cp.Minimize(c.T * x - cp.sum(cp.log(x)))
    prob = cp.Problem(obj, constraints)
    solver_return = prob.solve()

    print(f"implemented method optimal value: {(c.T @ x_optimal)[0][0] - np.sum(np.log(x_optimal))}")
    print(f"CVXPY optimal value: {prob.value}")

    log_plot(
        np.arange(len(lambda_hist)),
        lambda_hist,
        title="lambda history plot",
        x_label="iteration number",
        y_label="lambda ^ 2 / 2"
    )
