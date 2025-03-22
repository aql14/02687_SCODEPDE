# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 12:14:26 2025

@author: alvar
"""

import numpy as np
from scipy.sparse import kron, eye, diags
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import scipy as sp

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
    return -32 * pi**2 * np.sin(4 * pi * (x + y)) - 16 * pi**2 * (x**2 + y**2) * np.cos(4 * pi * x * y)
    # return 0 + 0*x*y  # Should be computed properly for actual test cases

# Exact solution
def exact_solution(x, y):
    pi = np.pi
    return np.sin(4 * pi * (x + y)) + np.cos(4 * pi * x * y)
    # return 1 + 1*x

# Apply boundary conditions (vectorized) ADAPT TO 9 LAPLACIAN
def f_BC_9pt(f, h, u_exact):
    main_coeff = 4/(6*h**2)
    diag_coeff = 1/(6*h**2)
    
    # Left boundary
    f[1:m+1, 1] -= main_coeff * u_exact[1:m+1, 0]
    f[1:m+1, 1] -= diag_coeff * (u_exact[0:m, 0] + u_exact[2:m+2, 0])
    
    # Right boundary
    f[1:m+1, m] -= main_coeff * u_exact[1:m+1, m+1]
    f[1:m+1, m] -= diag_coeff * (u_exact[0:m, m+1] + u_exact[2:m+2, m+1])
    
    # Bottom boundary
    f[1, 1:m+1] -= main_coeff * u_exact[0, 1:m+1]
    f[1, 1:m+1] -= diag_coeff * (u_exact[0, 0:m] + u_exact[0, 2:m+2])
    
    # Top boundary
    f[m, 1:m+1] -= main_coeff * u_exact[m+1, 1:m+1]
    f[m, 1:m+1] -= diag_coeff * (u_exact[m+1, 0:m] + u_exact[m+1, 2:m+2])
    
    # Corner terms
    f[1,1] += diag_coeff * u_exact[0,0]
    f[m,m] += diag_coeff * u_exact[m+1,m+1]
    f[1,m] += diag_coeff * u_exact[0,m+1]
    f[m,1] += diag_coeff * u_exact[m+1,0]

    return f

# Compute Laplacian of f for deferred correction
def laplacian_f(x,y):
    pi = np.pi
    return 64*pi**2*(4*pi**2*x**4*np.cos(4*pi*x*y) + 8*pi**2*x**2*y**2*np.cos(4*pi*x*y) + 8*pi*x*y*np.sin(4*pi*x*y) + 4*pi**2*y**4*np.cos(4*pi*x*y) + 16*pi**2*np.sin(pi*(4*x + 4*y)) - np.cos(4*pi*x*y))

# Mesh
m = 51
h = 1 / (m + 1)
x = np.linspace(0, 1, m+2)
y = np.linspace(0, 1, m+2)
X, Y = np.meshgrid(x, y)

# Compute exact solution and Laplacian
u_exact = exact_solution(X, Y)
exact_laplacian = analytical_laplacian(X, Y)

# Construct right-hand side (vectorized)
f = exact_laplacian.copy()

# Apply boundary conditions
f_complete = f_BC_9pt(f, h, u_exact)


# Apply deferred correction
f_corrected = f_complete + (h**2 / 12) * laplacian_f(X, Y)

# Extract only the interior points
f = f_corrected[1:-1, 1:-1]

# Construct the matrix A
sparse_A = poisson9(m)

# Flatten to match dimensionality
f_flatten = f.flatten('F')

# Solve system
u_app = sp.sparse.linalg.spsolve(sparse_A, f_flatten)

# Fill the full grid with computed values
U = u_exact.copy()
U[1:-1, 1:-1] = u_app.reshape((m, m), order='F')

# Compute error
error = np.abs(U - u_exact)

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
ax2.set_title('Solution U Contour: Error')

# Save figur
# fig.savefig('2b_c_Solution.png', bbox_inches='tight', dpi=200)

#plt.tight_layout()

plt.show()