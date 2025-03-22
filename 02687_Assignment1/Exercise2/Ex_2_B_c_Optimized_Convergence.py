# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 10:49:51 2025

@author: alvar
"""

import numpy as np
from scipy.sparse import kron, eye, diags
import matplotlib.pyplot as plt
import scipy as sp

eps = 2.220446049250313e-16

def poisson5(m):
    e = np.ones(m)
    S = diags([e, -2*e, e], [-1, 0, 1], shape=(m, m))
    I = eye(m)
    A = kron(I, S) + kron(S, I)
    A = (m + 1) ** 2 * A
    return A

# Laplacian of the exact solution
def analytical_laplacian(x, y):
    pi = np.pi
    return -32 * pi**2 * np.sin(4 * pi * (x + y)) - 16 * pi**2 * (x**2 + y**2) * np.cos(4 * pi * x * y)

# Exact solution
def exact_solution(x, y):
    pi = np.pi
    return np.sin(4 * pi * (x + y)) + np.cos(4 * pi * x * y)

# Include BC in the right-hand side f
def f_BC(f,h,u_exact):
    f[1:m+1, 1] = f[1:m+1, 1] - u_exact[1:m+1, 0] / h**2   # Left boundary
    f[1:m+1, m] = f[1:m+1, m] - u_exact[1:m+1, m+1] / h**2 # Right boundary
    f[1, 1:m+1] = f[1, 1:m+1] - u_exact[0, 1:m+1] / h**2   # Bottom boundary
    f[m, 1:m+1] = f[m, 1:m+1] - u_exact[m+1, 1:m+1] / h**2 # Top boundary
    return f

s = np.arange(3,10+1,1)
h_values = 1/(2**s)
m_values = (1/h_values - 1).astype('int')

# Initialize max norm error for each h_value
error_total = np.zeros(len(h_values))

for i, h in enumerate(h_values):
    m = (1/h - 1).astype('int')
    x = np.linspace(0, 1, m+2)
    y = np.linspace(0, 1, m+2)
    
    X, Y = np.meshgrid(x, y)
    
    # Compute exact solution and Laplacian
    u_exact = exact_solution(X, Y)
    exact_laplacian = analytical_laplacian(X, Y)

    # Construct right-hand side (vectorized)
    f = exact_laplacian.copy()
    
    # Extract only the interior points
    f_complete = f_BC(f,h,u_exact) 
    f = f_complete[1:-1, 1:-1]
    
    # Construct the matrix A
    sparse_A = poisson5(m)
    #A_array = sparse_A.toarray()
    
    # Flatten to match dimensionality
    f_flatten = f.flatten('F')
    
    # Solve system
    u_app = sp.sparse.linalg.spsolve(sparse_A, f_flatten)
    
    # Fill the full grid with computed values
    U = u_exact.copy()
    U[1:-1, 1:-1] = u_app.reshape((m, m), order='F')
    
    # Compute error
    error = U[1:-1,1:-1] - u_exact[1:-1,1:-1]

    # Total error
    # error_total[i] = np.max(np.abs(error))
    error_total[i] = np.linalg.norm((U - u_exact).flatten(), np.inf)
    
fig = plt.figure(figsize=(6,4),dpi=200)
axes = plt.gca()

axes.loglog(h_values, h_values**1, linestyle='--', label=r'$\mathcal{O}(h)$',color='slategrey')
axes.loglog(h_values, h_values**2, linestyle='--', label=r'$\mathcal{O}(h^2)$', color='lightsteelblue')
axes.loglog(h_values, h_values**3, linestyle='--', label=r'$\mathcal{O}(h^3)$', color='steelblue')
axes.loglog(h_values, error_total, linestyle='-', label=r'$||\tau||_{\infty}$', color='firebrick')
axes.set_xlabel(r'$log(h)$',fontsize=12)
axes.set_ylabel(r'$log(||\tau||)$',fontsize=12)
axes.set_title('Convergence Test',fontsize=14)
axes.legend()
axes.grid(True, linestyle="--",linewidth=0.5)
plt.tight_layout()

#fig.savefig('2b_c_Convergence.png', bbox_inches='tight', dpi=200)