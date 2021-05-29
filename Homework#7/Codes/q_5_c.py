from q_5_b import solve_lp_barrier_method
import numpy as np
import cvxpy as cp


def solve_lp(
        A,
        b,
        c
):
    m, n = A.shape
    x_0 = np.linalg.lstsq(A, b)[0].reshape(n, 1)
    t_0 = 2 + np.max([0, -1 * np.min(x_0)])

    A_prime = np.concatenate([
        A,
        -1 * A @ np.ones((n, 1))
    ], axis=1)
    b_prime = b - A @ np.ones((n, 1))
    z_0 = x_0 + t_0 * np.ones_like(x_0) - np.ones_like(x_0)
    # print(f"@@@@@@: {x_0.shape}").
    c_prime = np.concatenate([
        np.zeros_like(x_0),
        np.ones((1, 1))
    ], axis=0)
    # print(f"@@@@@@: {z_0.shape}")
    x_0 = np.concatenate([
        z_0,
        t_0.reshape(1, 1)
    ], axis=0)
    z_optimal, gap, sub_optimality_gap_hist, newton_iters_hist = solve_lp_barrier_method(
        A_prime,
        b_prime,
        c_prime,
        x_0
    )
    part_1_n_newton_steps = np.sum(newton_iters_hist)
    if z_optimal is None:
        print("ERROR: Problem is infeasible!!!")
        return {
            'optimal_value': None,
            'x_optimal': None,
            'gap': None,
            'status': 'Infeasible',
            'newton_steps': None
        }

    x_0 = z_optimal[:n] - z_optimal[n][0] * np.ones((n, 1)) + np.ones((n, 1))

    x_optimal, gap, sub_optimality_gap_hist, newton_iters_hist = solve_lp_barrier_method(
        A,
        b,
        c,
        x_0
    )
    part_2_n_newton_steps = np.sum(newton_iters_hist)
    if x_optimal is None:
        return None

    optimal_value = (c.T @ x_optimal)[0][0]
    return {
        'optimal_value': optimal_value,
        'x_optimal': x_optimal,
        'gap': gap,
        'status': 'Optimal',
        'newton_steps': part_1_n_newton_steps + part_2_n_newton_steps
    }


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
    x_0 = np.random.rand(n, 1) + 0.1
    b = A @ x_0
    c = np.random.rand(n, 1)

    solver_dict = solve_lp(
        A,
        b,
        c
    )
    print(solver_dict)

    x = cp.Variable((n, 1))
    constraints = [
        A * x == b,
        x >= 0
    ]
    obj = cp.Minimize(c.T * x)
    prob = cp.Problem(obj, constraints)
    solver_return = prob.solve()

    print(f"optimal value from barrier method: {solver_dict['optimal_value']}")
    print(f"optimal value from CVXPY solver: {prob.value}")
