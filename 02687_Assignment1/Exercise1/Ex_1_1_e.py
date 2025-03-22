# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 09:27:32 2025

@author: niroj
"""

import numpy as np
import math as mt
import matplotlib.pyplot as plt


def kron_del(i, j):
    return 1 if i == j else 0

def f(x):
    return np.exp(np.cos(x))

def fdcoeffV(k,xbar,x):
    n = len(x)
    A = np.ones((n,n))
    xrow = (x-xbar).transpose()
    for i in range(2,n+1):
        A[i-1,:] = (xrow**(i-1)/mt.factorial(i-1))
    b = np.zeros((n,1))
    b[k] = 1
    c = np.linalg.solve(A, b)
    
    return c

def operator_derivative(x,c):
    result = 0
    for i in range(len(x)):
        result = result + c[i]*f(x[i])
    
    return result

def ddf(x):
    return np.exp(np.cos(x))*(-np.cos(x) +  np.sin(x)**2)

def Order_Accuarcy(k,x):
    N = 10
    Cn = np.zeros(N)
    for n in range(0, N):
        Cn[n] += -kron_del(n, k)*(1**(-n))
        for i in range(len(X_e)):
            Cn[n] +=  coef_e[i] * ((X_e[i])**n / mt.factorial(n))
        if np.abs(Cn[n])>=1e-10:
            order = n-k
            return order
    


xbar = 0
k = 0

# Testing: h = 1
X_e = np.arange(-0.5,1.5+1,1)
coef_e = fdcoeffV(k, xbar, X_e)

N = 10
Cn = np.zeros(N)

am = np.array([1, 2, 3, 4, 5])  # Example values
m = np.array([1, 2, 3, 4, 5])   # Example values

Cn = Order_Accuarcy(k, X_e)
N = 10
Cn = np.zeros(N)

for n in range(0, N):
    Cn[n] += -kron_del(n, k)*(1**(-n))
    for i in range(len(X_e)):
        Cn[n] +=  coef_e[i] * ((X_e[i])**n / mt.factorial(n))

s = np.arange(2,20+1,1)
h = 1/(2**s)

TE_e = np.zeros((len(h),1))


ddf_bar = ddf(xbar)

for i in range(len(h)):
    X_e_CT = np.arange(xbar-0.5*h[i],xbar+1.5*h[i] + h[i],h[i])
    coef_e_CT = fdcoeffV(k,xbar,X_e_CT)
    Cn = Order_Accuarcy(0, X_e_CT)
    f_e = operator_derivative(X_e_CT, coef_e_CT)
    TE_e[i] = np.abs(f(xbar)-f_e)

# Plot in log-log scale the THEROETICAL convergence test
plt.loglog(h, h**3, linestyle='--', label=r'$\mathcal{O}(h^3)$')
plt.loglog(h, h**4, linestyle='--', label=r'$\mathcal{O}(h^4)$')
plt.loglog(h, TE_e, linestyle='-', label=r'$\tau_{C}$')

# Labels and legend
plt.xlabel(r'$h$')
plt.ylabel(r'Truncation Error')
plt.title('Truncation Error in Log-Log Scale')
plt.xlim([1e-6,0])
plt.ylim([1e-26,0])
plt.legend()
plt.grid(True, linestyle="--")

# Show plot
plt.show()