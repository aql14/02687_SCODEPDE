# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 19:52:53 2025

@author: alvar
"""

import numpy as np
from scipy.sparse import kron, eye, diags
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import scipy as sp
import time

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

# Include BC in the right-hand side f (5-point Laplacian)
def f_BC_5p(f,u_exact):
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

# Under-relaxed Jacobi smoothing
def smooth(U,omega,nsmooth,F):

    # Initialize residuals
    residuals = []

    # Iteration parameters
    tolerance = 1e-8
    max_iterations = nsmooth

    # Mesh size
    m = int(np.sqrt(len(U)))
    h = 1 / (m + 1)

    # Initial residual
    residuals.append(np.linalg.norm(F + Amult(U)))

    # Iteration
    for k in range(1,max_iterations):

        # Update the solution
        U_new = U + omega * h**2 /4 * ( -Amult(U) - F)

        # Calculate the residual
        residuals.append(np.linalg.norm(F + Amult(U_new)))

        # Stopping criterion
        if residuals[k]/residuals[0] < tolerance:
            print(f"Converged in {k+1} iterations with final residual {residuals[k]:.2e}")
            return U_new
            break
        
        U = U_new

    print(f"Converged in {k+1} iterations with final residual {residuals[k]:.2e}")
    return U

def Amult(U):
    m = int(np.sqrt(len(U)))
    h = 1/(m+1)
    
    # Reshape U into a 2D array for easier indexing
    U = U.reshape((m, m), order='F')
    
    # Create expanded U to allow slicing
    expanded_U = np.zeros((m+2,m+2))
    expanded_U[1:-1,1:-1] = U
    
    # Initialize product array
    AU = np.zeros_like(U)
    
    # Stencil terms
    left_term   = expanded_U[1:-1, :-2]
    right_term  = expanded_U[1:-1, 2:]   
    bottom_term = expanded_U[:-2, 1:-1]  
    top_term    = expanded_U[2:, 1:-1]   
    self_term   = -4*U                 

    #Solution
    AU = (left_term+right_term+bottom_term+top_term+self_term)/h**2  
    
    return -AU.flatten('F')

# Interpolate function: coarse grid to fine grid
def interpolate(e_coarse,m):
    
    # Grid properties
    m_coarse = int(np.sqrt(len(e_coarse)))
    h_coarse = 1 / (m_coarse + 1)   
    

    # Project back onto the fine grid
    # We add the boundary nodes for easier indexing
    e = np.zeros((m+2,m+2))
    e[2:-2:2,2:-2:2] = e_coarse.reshape((m_coarse, m_coarse), order='F')

    # INTERPOLATION
    
    # First Sweep: Horizontal
    e[2:-2:2, 1:-1:2] = 1/2 * (e[2:-2:2, 0:-1:2] + e[2:-2:2, 2::2])
    
    # Second Sweep: Vertical
    e[1:-1:2, 2:-2:2] = 1/2 * (e[0:-1:2, 2:-2:2] + e[2::2, 2:-2:2])
    
    # Third Sweep: Bilinear Interpolation
    e[1:-1:2, 1:-1:2] = 1/4 * (e[1:-1:2, 0:-2:2] + e[1:-1:2, 2::2] + e[0:-2:2, 1:-1:2] + e[2::2, 1:-1:2])
    
    # Get only the interior points back
    e = e[1:-1,1:-1]
        
    return e.flatten('F')

# Coarsen function: fine grid to coarse grid
def coarsen(r):
    m = int(np.sqrt(len(r)))
    r = r.reshape((m, m), order='F')
    m_coarse = int((m - 1) / 2)
    
    # Direct Interpolation
    # r_coarse = r[1::2,1::2]      # Pick each two elements from the first one
    
    # Full-Weighting Interpolation
    # r_coarse = (
    #     1/16 * r[0:-2:2, 0:-2:2] + 1/8 * r[1:-1:2, 0:-2:2] + 1/16 * r[2::2, 0:-2:2] +
    #     1/8  * r[0:-2:2, 1:-1:2] + 1/4  * r[1:-1:2, 1:-1:2] + 1/8  * r[2::2, 1:-1:2] +
    #     1/16 * r[0:-2:2, 2::2] + 1/8 * r[1:-1:2, 2::2] + 1/16 * r[2::2, 2::2]
    # )
    
    r_coarse = (
        1/16 * r[:-2:2,:-2:2] + 1/8 * r[:-2:2,1:-1:2] + 1/16 * r[:-2:2,2::2] +
        1/8 * r[1:-1:2,:-2:2] + 1/4 * r[1:-1:2,1:-1:2] + 1/8 * r[1:-1:2,2::2] +
        1/16 * r[2::2,:-2:2] + 1/8 * r[2::2,1:-1:2] + 1/16 * r[2::2,2::2]
        )
    
    return r_coarse.flatten('F')

# Recursive V-Cycle Solver
def Vcycle(U,omega,nsmooth,m,F):
    
    h = 1.0/(m+1)
    
    if m == 1:
        # We are at the coarsest level (1 node)
        
        # Create coarse grid matrix
        # A_coarse = poisson5(m)
        A_coarse = -4/(h**2)

        # Solve the coarse problem
        e_coarse = F/A_coarse
        
        U = e_coarse
        
    else:
        
        # 1. Pre-Smoothing
        U = smooth(U, omega, nsmooth, F)
        
        # 2. Compute Residual
        r = F + Amult(U)
        
        # 3. Coarsen Residual
        r_coarse = coarsen(r)
        
        # 4. Recurse to Vcycle
        mc = int(np.sqrt(len(r_coarse)))
        e_coarse = Vcycle(np.zeros_like(r_coarse), omega, nsmooth, mc, -r_coarse)
        
        # 5. Interpolate Error
        e = interpolate(e_coarse, m)
        
        # 6. Update Solution
        U -= e
        
        # 7. Post-Smoothing
        U = smooth(U, omega, nsmooth, F)    
    
    return U

# Mesh
m = 2**10-1
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
f_complete = f_BC_5p(f,u_exact) 
f = f_complete[1:-1, 1:-1]

# Construct the matrix A

# Flatten to match dimensionality
f_flatten = f.flatten('F')

# Solve system
u_app = np.zeros(m**2)

# u_app = smooth(u_app, 2/3, 10000, f_flatten)

# Vcycle
u_app = Vcycle(u_app, 2/3, 10, m, f_flatten)

# Fill the full grid with computed values
U = u_exact.copy()
U[1:-1, 1:-1] = u_app.reshape((m, m), order='F')

# Compute error
error = np.abs(U - u_exact)

# print(np.linalg.norm(error,np.inf))
print(np.linalg.norm((U - u_exact).flatten(), np.inf))

# t2 = time.perf_counter()


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

plt.show()