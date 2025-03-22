# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:12:07 2025

@author: alvar
"""

import numpy as np
from scipy.sparse import kron, eye, diags
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import scipy as sp
import time

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
    # return 0 + 0*x*y
    return -32 * pi**2 * np.sin(4 * pi * (x + y)) - 16 * pi**2 * (x**2 + y**2) * np.cos(4 * pi * x * y)

# Exact solution
def exact_solution(x, y):
    pi = np.pi
    # return 1+ 1*x
    return np.sin(4 * pi * (x + y)) + np.cos(4 * pi * x * y)

# Include BC in the right-hand side f
def f_BC_5p(f):
    # 5 point nodes coefficients
    main_coeff = 1/h**2
    # Left boundary
    f[1:m+1, 1] = f[1:m+1, 1] - main_coeff * u_exact[1:m+1, 0]
    # Right boundary
    f[1:m+1, m] = f[1:m+1, m] - main_coeff * u_exact[1:m+1, m+1]
    # Bottom boundary
    f[1, 1:m+1] = f[1, 1:m+1] - main_coeff * u_exact[0, 1:m+1]
    # Top boundary
    f[m, 1:m+1] = f[m, 1:m+1] - main_coeff * u_exact[m+1, 1:m+1]
    
    return f

# Mesh
t1 = time.perf_counter()

m = 2**8-1
h = 1 / (m + 1)
x = np.linspace(0, 1, m+2)
y = np.linspace(0, 1, m+2)

X, Y = np.meshgrid(x, y)

# Compute exact solution and Laplacian
u_exact = exact_solution(X, Y)
exact_laplacian = analytical_laplacian(X, Y)

# Construct right-hand side (vectorized)
f = exact_laplacian.copy()

# Extract only the interior points
f_complete = f_BC_5p(f) 
f = f_complete[1:-1, 1:-1]



# Construct the matrix A
sparse_A = poisson5(m)
# A_array = sparse_A.toarray()

# Flatten to match dimensionality
f_flatten = f.flatten('F')

# Solve system
# u_app = np.linalg.solve(A_array, f_flatten)
u_app = sp.sparse.linalg.spsolve(sparse_A, f_flatten)

# Fill the full grid with computed values
U = u_exact.copy()
U[1:-1, 1:-1] = u_app.reshape((m, m), order='F')

# Compute error
error = np.abs(U - u_exact)

t2 = time.perf_counter()

print(np.linalg.norm(error,np.inf))



print(f'Computation Time is: {t2-t1}')


# Create figure with 2 columns
fig = plt.figure(figsize=(12, 4), dpi=100)
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1],wspace=0.2)

# ----- Subplot 1: 3D Surface Plot -----
ax1 = fig.add_subplot(gs[0], projection='3d')
ax1.plot_surface(X, Y, U, cmap='viridis')

# Reverse axes for MATLAB-like orientation
ax1.set_xlim(ax1.get_xlim()[::-1])  
ax1.set_ylim(ax1.get_ylim()[::-1])  

# Set view to match MATLAB
ax1.view_init(elev=30, azim=90-37.5)

# Labels
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('u(X,Y)')
ax1.set_title('Surface plot of u(x,y)')



# ----- Subplot 2: Contour Plot -----
ax2 = fig.add_subplot(gs[1])
contour = ax2.contourf(X, Y, error, levels=50, cmap='viridis')
fig.colorbar(contour, ax=ax2)  # Add colorbar
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Solution U Contour')

# Save figur
# fig.savefig('2b_c_Solution.png', bbox_inches='tight', dpi=200)

#plt.tight_layout()

plt.show()