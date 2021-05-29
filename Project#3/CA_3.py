#!/usr/bin/env python
# coding: utf-8

# ### Import libraries

# In[22]:


from robust_powe_data import *
import cvxpy as cp
import warnings
warnings.filterwarnings('ignore')


# In[23]:


u = u.reshape(-1, 1)
c = c.reshape(-1, 1)


# ## First part
# We Supposed that all plants are online

# ##### The objective of problem is:
# $$
# \operatorname{tr} L^{T} A
# $$

# ##### The constraints are:
# 1) $$
# P^{\mathrm{ar}} \succeq u
# $$
# 2) $$
# P^{\mathrm{tr}} \preceq c
# $$
# 3) $$
# P^{\mathrm{ar}} \succeq 0
# $$
# 4) $$
# P^{\mathrm{tr}} \succeq 0
# $$
# 5) $$
# P_{i j}^{\mathrm{tr}} \geq P_{i j}^{\mathrm{ar}}+\alpha \frac{L_{i j}\left(P_{i j}^{\mathrm{ar}}\right)^{2}}{A_{i j}}, \quad i=1, \ldots, m, j=1, \ldots, n
# $$
# 
# ###### I have used quad_over_lin function from cvxpy for the last constraint because dcp rules didn't hold without it.

# In[24]:


n_ones = np.ones((n, 1))
m_ones = np.ones((m, 1))

P_tr = cp.Variable((m,n))
P_ar = cp.Variable((m,n))
A = cp.Variable((m,n))

constraints = [
               (m_ones.T @ P_ar).T >= u,
               P_tr @ n_ones <= c,
               P_ar >= 0,
               P_tr >= 0
]

constraints += [
                P_tr[i][j] - P_ar[i][j] - alpha * L[i][j] * cp.quad_over_lin(P_ar[i][j], A[i][j]) >= 0 for i in range(m) for j in range(n)
]

prob = cp.Problem(cp.Minimize(cp.trace(L.T @ A)),
                  constraints)
prob.solve()


# ##### The number of elements more than $10^{-3}$ is reported here:

# In[25]:


A_prime = np.abs(A.value.reshape(-1,)) > 1e-3
print(f"the number of elements more than 1e-3 is: {sum(val for val in A_prime)}")


# ##### The total cost is reported here:

# In[ ]:


print(f"the total cost is: {round(prob.value, 2)}")


# ## Second Part
# ### Now we model our problem more realistic!

# We supposed that each destination has some offline supplier.

# In[ ]:


k_variations = [1, 2, 3, 4]


# ##### The additional constraint is:
# $$
# \sum_{i=1}^{m} O_{i} P_{i j}^{\mathrm{ar}} \geq u_{j} \quad \text { for all } O \in \mathcal{O}_{k}
# $$

# ###### I have used quad_over_lin function from cvxpy for the additional constraint because dcp rules didn't hold without it. Its functionality is the summation over n minimum elements of the vector.

# ##### The number of elements more than $10^{-3}$ and the minimum cost for each variation of k are reported here:

# for k in k_variations:
#     constraints = [
#                   (m_ones.T @ P_ar).T >= u,
#                   P_tr @ n_ones <= c,
#                   P_ar >= 0,
#                   P_tr >= 0
#     ]
# 
#     constraints += [
#                     P_tr[i][j] - P_ar[i][j] - alpha * L[i][j] * cp.quad_over_lin(P_ar[i][j], A[i][j]) >= 0 for i in range(m) for j in range(n)
#     ]
# 
#     constraints += [
#                     cp.sum_smallest(P_ar.T[i], m - k) >= u[i] for i in range(n)
#     ]
# 
#     prob = cp.Problem(cp.Minimize(cp.trace(L.T @ A)),
#                       constraints)
#     prob.solve()
# 
#     print(f"########################## k = {k} ###############################")
#     A_prime = np.abs(A.value.reshape(-1,)) > 1e-3
#     print(f"the number of elements more than 1e-3 is: {sum(val for val in A_prime)}")
#     print(f"the total cost is: {round(prob.value, 2)}")

# In[26]:


print(f"the total cost is: {round(prob.value, 2)}")


# ## Second Part
# ### Now we model our problem more realistic!

# We supposed that each destination has some offline supplier.

# In[27]:


k_variations = [1, 2, 3, 4]


# ##### The additional constraint is:
# $$
# \sum_{i=1}^{m} O_{i} P_{i j}^{\mathrm{ar}} \geq u_{j} \quad \text { for all } O \in \mathcal{O}_{k}
# $$

# ###### I have used quad_over_lin function from cvxpy for the additional constraint because dcp rules didn't hold without it. Its functionality is the summation over n minimum elements of the vector.

# ##### The number of elements more than $10^{-3}$ and the minimum cost for each variation of k are reported here:

# In[28]:


for k in k_variations:
    constraints = [
                  (m_ones.T @ P_ar).T >= u,
                  P_tr @ n_ones <= c,
                  P_ar >= 0,
                  P_tr >= 0
    ]

    constraints += [
                    P_tr[i][j] - P_ar[i][j] - alpha * L[i][j] * cp.quad_over_lin(P_ar[i][j], A[i][j]) >= 0 for i in range(m) for j in range(n)
    ]

    constraints += [
                    cp.sum_smallest(P_ar.T[i], m - k) >= u[i] for i in range(n)
    ]

    prob = cp.Problem(cp.Minimize(cp.trace(L.T @ A)),
                      constraints)
    prob.solve()

    print(f"########################## k = {k} ###############################")
    A_prime = np.abs(A.value.reshape(-1,)) > 1e-3
    print(f"the number of elements more than 1e-3 is: {sum(val for val in A_prime)}")
    print(f"the total cost is: {round(prob.value, 2)}")


# ### Inference part:
# 
# Since we decrease the number of online plants, we have to increase the size of transmission pipelines. This augmentation causes more cost. The hypothesize was confirmed by the experiments.
