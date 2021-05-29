import cvxpy as cp
from microgrid_data import *

# import cvxpy as cp

p_batt = cp.Variable(N)
p_buy = cp.Variable(N)
p_sell = cp.Variable(N)
p_grid = p_buy - p_sell
q = cp.Variable(N)

obj = cp.Minimize(
    (1/4) * p_buy.T * R_buy - (1/4) * p_sell * R_sell
)

constraints = [
              p_ld == p_grid + p_batt + p_pv,
              q >= 0,
              q <= Q,
              q[0] == q[-1] - (1/4) * p_batt[-1],
              p_batt >= -C,
              p_batt <= D,
              p_buy >= 0,
              p_sell >= 0
] + [
     q[i+1] == q[i] - (1/4) * p_batt[i] for i in range(95)
]

prob = cp.Problem(obj, constraints)
prob.solve(solver=cp.ECOS)
print(f"Minimum cost is: {obj.value}")

q = q.value
p_grid = p_grid.value
p_batt = p_batt.value

plt.figure(figsize=fig_size)
plt.plot(np.arange(N), p_grid)
plt.title('p_grid (kW)', fontsize=19)
plt.ylabel('Power (kW)')
plt.xticks(xtick_vals, xtick_labels)
plt.show()

plt.figure(figsize=fig_size)
plt.plot(np.arange(N), p_ld)
plt.title('p_ld (kW)', fontsize=19)
plt.ylabel('Power (kW)')
plt.xticks(xtick_vals, xtick_labels)
plt.show()

plt.figure(figsize=fig_size)
plt.plot(np.arange(N), p_pv)
plt.title('p_pv (kW)', fontsize=19)
plt.ylabel('Power (kW)')
plt.xticks(xtick_vals, xtick_labels)
plt.show()

plt.figure(figsize=fig_size)
plt.plot(np.arange(N), p_batt)
plt.title('p_batt (kW)', fontsize=19)
plt.ylabel('Power (kW)')
plt.xticks(xtick_vals, xtick_labels)
plt.show()

plt.figure(figsize=fig_size)
plt.plot(np.arange(N), q)
plt.title('Energy (kWh)', fontsize=19)
plt.ylabel('Battery Charge (kWh)')
plt.xticks(xtick_vals, xtick_labels)
plt.show()

dual_vals = -constraints[0].dual_value
LMP = 4 * dual_vals

plt.figure(figsize=fig_size)
plt.plot(R_buy, '--', label='Buy Price')
plt.plot(R_sell, '--', label='Sell Price')
plt.plot(LMP, label='LMP')
plt.xlabel('Time')
plt.ylabel('Price ($/kWh)')
plt.title('Locational Marginal Price', fontsize=19)
plt.legend()
plt.xticks(xtick_vals, xtick_labels)
plt.show()

load_cost = p_ld @ dual_vals
batt_cost = -p_batt @ dual_vals
pv_cost = -p_pv @ dual_vals
grid_cost = p_grid @ dual_vals

print(f"Load cost is: {round(load_cost, 2)}")
print(f"Battery cost is: {round(batt_cost, 2)}")
print(f"Effective grid cost is: {round(grid_cost, 2)}")

net_cost = -load_cost - batt_cost - pv_cost + grid_cost
print(f"Net cost of grid is: {round(net_cost, 2)}")