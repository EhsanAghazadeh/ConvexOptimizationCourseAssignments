from q_5_a import solve_lp_newton_method
import numpy as np
from utils import *
import cvxpy as cp

def solve_lp_barrier_method(
        A,
        b,
        c,
        x_0,
        t_0=1,
        mu=10,
        epsilon=1e-3
):
    x = x_0
    t = float(t_0)
    sub_optimality_gap_hist = []
    newton_iters_hist = []
    n = len(x_0)

    while True:
        x_optimal, nu_optimal, lambda_hist = solve_lp_newton_method(
            A,
            b,
            t * c,
            x
        )
        if x_optimal is None:
            print("ERROR: centering step didn't work.")
            return None, None, None
        x = x_optimal
        gap = n / t
        sub_optimality_gap_hist.append(gap)
        newton_iters_hist.append(len(lambda_hist))
        if gap < epsilon:
            break
        t = mu * t

    return x, gap, sub_optimality_gap_hist, newton_iters_hist


if __name__ == '__main__':
    mu = 10
    t_0 = 1
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
    x_0 = np.random.rand(n, 1) + 0.1
    b = A @ x_0
    c = np.random.rand(n, 1)

    x_optimal, last_gap, sub_optimality_gap_hist, newton_iters_hist = solve_lp_barrier_method(
        A,
        b,
        c,
        x_0,
        mu=20
    )
    print("optimal x is:")
    print(x_optimal)
    print(f"cumulative {newton_iters_hist}")
    cumulative_hist = np.cumsum(newton_iters_hist)
    print(f"cumulative {cumulative_hist}")

    x = cp.Variable((n, 1))
    constraints = [
        A * x == b,
        x >= 0
    ]
    obj = cp.Minimize(c.T * x)
    prob = cp.Problem(obj, constraints)
    solver_return = prob.solve()

    print(f"optimal value from barrier method: {(c.T @ x_optimal)[0][0]}")
    print(f"optimal value from CVXPY solver: {prob.value}")

    log_plot(
        np.arange(cumulative_hist[-1]),
        np.array([sub_optimality_gap_hist[idx] for idx, val in enumerate(newton_iters_hist) for i in range(val)]),
        title="Number of steps Vs Gap",
        x_label="newton iteration number",
        y_label="gap"
    )

    log_plot(
        cumulative_hist,
        sub_optimality_gap_hist,
        title="Number of steps Vs Gap",
        x_label="cumulative number of steps",
        y_label="gap"
    )
