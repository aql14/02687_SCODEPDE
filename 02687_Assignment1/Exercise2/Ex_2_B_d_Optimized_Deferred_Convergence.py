# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 12:20:45 2025

@author: alvar
"""

import numpy as np
from scipy.sparse import kron, eye, diags
import matplotlib.pyplot as plt
import scipy as sp

eps = 2.220446049250313e-16

def poisson9(m):
    e = np.ones(m)
    S = diags([-e, -10*e, -e], [-1, 0, 1], shape=(m, m))
    I = diags([-1/2*e, e, -1/2*e], [-1, 0, 1], shape=(m, m))
    A = kron(I, S) + kron(S, I)
    A = 1/6*(m + 1) ** 2 * A
    return A

# Laplacian of the exact solution
def analytical_laplacian(x, y):
    pi = np.pi
    # return 0 + 0*x*y
    return -32 * pi**2 * np.sin(4 * pi * (x + y)) - 16 * pi**2 * (x**2 + y**2) * np.cos(4 * pi * x * y)

# Exact solution
def exact_solution(x, y):
    pi = np.pi
    # return 1+ 1*x
    return np.sin(4 * pi * (x + y)) + np.cos(4 * pi * x * y)

# Compute Laplacian of f for deferred correction
def laplacian_f(x,y):
    pi = np.pi
    return 64*pi**2*(4*pi**2*x**4*np.cos(4*pi*x*y) + 8*pi**2*x**2*y**2*np.cos(4*pi*x*y) + 8*pi*x*y*np.sin(4*pi*x*y) + 4*pi**2*y**4*np.cos(4*pi*x*y) + 16*pi**2*np.sin(pi*(4*x + 4*y)) - np.cos(4*pi*x*y))

    
# Apply boundary conditions (vectorized) ADAPT TO 9 LAPLACIAN
def f_BC_9pt(f, h, u_exact):
    
    # 9-point nodes coefficients
    main_coeff = 4/(6*h**2)
    diag_coeff = 1/(6*h**2)
    
    # Left boundary
    f[1:m+1, 1] = f[1:m+1, 1] - main_coeff * u_exact[1:m+1, 0]
    f[1:m+1, 1] = f[1:m+1, 1] - diag_coeff * (u_exact[0:m, 0] + u_exact[2:m+2, 0])
    
    # Right boundary
    f[1:m+1, m] = f[1:m+1, m] - main_coeff * u_exact[1:m+1, m+1]
    f[1:m+1, m] = f[1:m+1, m] - diag_coeff * (u_exact[0:m, m+1] + u_exact[2:m+2, m+1])
    
    # Bottom boundary
    f[1, 1:m+1] = f[1, 1:m+1] - main_coeff * u_exact[0, 1:m+1]
    f[1, 1:m+1] = f[1, 1:m+1] - diag_coeff * (u_exact[0, 0:m] + u_exact[0, 2:m+2])
    
    # Top boundary
    f[m, 1:m+1] = f[m, 1:m+1] - main_coeff * u_exact[m+1, 1:m+1]
    f[m, 1:m+1] = f[m, 1:m+1] - diag_coeff * (u_exact[m+1, 0:m] + u_exact[m+1, 2:m+2])
    
    # # TERM IN THE CORNERS
    f[1,1] += diag_coeff * u_exact[0,0]
    f[m,m] += diag_coeff * u_exact[m+1,m+1]
    f[1,m] += diag_coeff * u_exact[0,m+1]
    f[m,1] += diag_coeff * u_exact[m+1,0]

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
    f_complete = f_BC_9pt(f, h, u_exact)
    f_corrected = f_complete + (h**2 / 12) * laplacian_f(X,Y)
    f = f_corrected[1:-1, 1:-1]
    
    # Construct the matrix A
    sparse_A = poisson9(m)
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
axes.loglog(h_values, h_values**4, linestyle='--', label=r'$\mathcal{O}(h^4)$', color='navy')
axes.loglog(h_values, error_total, linestyle='-', label=r'$||\tau||_{\infty}$', color='firebrick')
axes.set_xlabel(r'$log(h)$',fontsize=12)
axes.set_ylabel(r'$log(||\tau||)$',fontsize=12)
axes.set_title('Convergence Test',fontsize=14)
axes.legend()
axes.grid(True, linestyle="--",linewidth=0.5)
plt.tight_layout()

#fig.savefig('2b_c_Convergence.png', bbox_inches='tight', dpi=200)