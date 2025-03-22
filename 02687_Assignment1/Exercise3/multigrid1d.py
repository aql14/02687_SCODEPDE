# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 18:33:46 2025

@author: alvar
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def coarsen(r):
    m = len(r)
    m_coarse = (m - 1) // 2
    h_coarse = 1 / (m_coarse + 1)
    r_coarse = r[1::2]      # Pick each two elements from the first one
    # r_coarse = 1/4 * r[:-2:2] + 1/2 * r[1:-1:2] + 1/4 * r[2::2]
    assert len(r_coarse) == m_coarse  # gives an error if this does not apply
    
    return r_coarse

def interpolate(r_coarse,m):
    

    # Coarse grid
    m_coarse = len(r_coarse)
    h_coarse = 1 / (m_coarse + 1)
    
    # Create coarse grid matrix
    A_coarse_diag_main = np.full(m_coarse, -2 / h_coarse**2)
    A_coarse_diag_off = np.full(m_coarse - 1, 1 / h_coarse**2)
    A_coarse = sp.sparse.diags([A_coarse_diag_off, A_coarse_diag_main, A_coarse_diag_off], offsets=[-1, 0, 1], format='csc')

    # Solve the coarse problem
    e_coarse = sp.sparse.linalg.spsolve(A_coarse, -r_coarse)

    # Project back onto the fine grid
    e = np.zeros(m)
    e[1::2] = e_coarse
        
    # Interpolation for even indices, ensuring boundary handling
    e[2:-1:2] = 0.5 * (e[1:-2:2] + e[3::2])  # Avoiding boundaries
    e[0] = 0.5 * e[1]  # Left boundary handling
    e[-1] = 0.5 * e[-2]  # Right boundary handling
        
    return e
        
a = .5

def psi(x):
    return 20*np.pi*x**3

def psidot(x):
    return 3*20*np.pi*x**2

def psiddot(x):
    return 2*3*20*np.pi*x

def f(x):
    return -20 + a*psiddot(x) * np.cos(psi(x)) - a*psidot(x)**2*np.sin(psi(x))

def u(x):
    return 1 + 12*x - 10*x**2 + a*np.sin(psi(x))

# Mesh
m = 155
h = 1/(m+1)

# Create sparse matrix A
diag_main = np.full(m, -2 / h**2)
diag_off = np.full(m - 1, 1 / h**2)
A = sp.sparse.diags([diag_off, diag_main, diag_off], offsets=[-1, 0, 1], format='csc')

X = np.linspace(0+h, 1-h, m)

# Right hand side with BC
F = f(X)
F[0] = F[0] - u(0)/h**2
F[-1] = F[-1] - u(1)/h**2

# Exact solution
Uhat = u(X)
Ehat = sp.sparse.linalg.spsolve(A,F) - Uhat

# Jacobi method setup
M = sp.sparse.diags(A.diagonal(), format='csc')
N = M - A
G = sp.sparse.linalg.spsolve(M, N)
b = sp.sparse.linalg.spsolve(M, F)

# Relaxation factor
omega = 2 / 3

# Initialize U2
U2 = 1 + 2 * X

# Iterative Jacobi method
for i in range(1, 11):
    U2 = (1 - omega) * U2 + omega * (G @ U2 + b)
    E2 = U2 - Uhat
    
# Calculate residual
r = F - A @ U2

# Coarsen the residual
r_coarse = coarsen(r)

# Interpolate
e = interpolate(r_coarse, m)

U2 = U2 - e
E2 = U2 - Uhat

print(np.linalg.norm(U2,np.inf))
print(np.linalg.norm(E2,np.inf))

# Smooth the error again --> Jacobi
for i in range(1, 11):
    U2 = (1 - omega) * U2 + omega * (G @ U2 + b)
    E2 = U2 - Uhat
    
print(np.linalg.norm(U2,np.inf))
print(np.linalg.norm(E2,np.inf))

plt.figure()
plt.plot(X,Uhat)
plt.plot(X,U2,'.')
plt.show