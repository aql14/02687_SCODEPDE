# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:09:03 2025

@author: alvar
"""

import numpy as np
import matplotlib.pyplot as plt

""" INITIAL GUESS """
def initial_guess(x):
    xbar = 0.5 * (a-b-alpha+beta)
    w_0 = 0.5 * (a+b-alpha-beta)
    return x - xbar + w_0*np.tanh(w_0*(x-xbar)/(2*eps))

""" G """
def G(u):
    N = len(u)
    G = np.zeros(N)
    for i in range(N):
        u_i = u[i]
        u_prev = u[i-1] if i>0 else alpha
        u_next = u[i+1] if i<N-1 else beta
    
        # Update G function
        G[i] = (eps/h**2) * (u_next-2*u_i+u_prev) + u_i * ((u_next-u_prev)/(2*h) - 1)
    
    return G

""" JACOBIAN """
def J(u):
    N = len(u)
    J = np.zeros((N,N))
    
    for i in range(N):
        u_i = u[i]
        u_prev = u[i-1] if i>0 else alpha
        u_next = u[i+1] if i<N-1 else beta
        
        # Diagonal
        J[i,i] = -2*eps/h**2 + 1/(2*h) * (u_next-u_prev) - 1
        
        # Off diagonal up
        if i < N-1:
            J[i,i+1] = eps/h**2 + u_i/(2*h)
        
        # Off diagonal down
        if i > 0:
            J[i,i-1] = eps/h**2 - u_i/(2*h)
        
    return J

""" Newton-Raphson """
def Newton_Raphson_Systems(G,J,u0,tol=1e-7,num_iter=100):
    u = u0.copy()
    iteration = 0
    
    while iteration < num_iter:
        jacobian = J(u)
        Fu = G(u)
        du = np.linalg.solve(jacobian, -Fu)
        
        u_new = u + du
        
        if np.linalg.norm(du, 2) < tol:            
            print(f"Newton-Raphson converged in {iteration+1} iterations.")
            return u_new
        
        iteration =+ 1
        u = u_new
        
    print("Newton-Raphson did not converge.")
    return u_new

# Parameters
eps = 0.1
alpha = -1
beta = 1.5

# Domain
a = 0
b = 1

""" Mesh Construction """
N = 20  # Interior points
h = (b-a)/(N+1)
xi = np.arange(a,b+h,h)

""" Approximate initial solution """
u0 = initial_guess(xi)[1:-1]

# Solution interior points
u_interior = Newton_Raphson_Systems(G, J, u0)
u_sol = np.concatenate(([alpha], u_interior, [beta]))

# Plot
fig = plt.figure()
axes = plt.gca()
axes.plot(xi,u_sol,color='firebrick')
axes.set_xlabel(r'$t$')
axes.set_ylabel(r'Solution $u(t)$')
axes.set_title('Non-linear BVP')
axes.grid(True, linestyle="--",linewidth=0.5)




