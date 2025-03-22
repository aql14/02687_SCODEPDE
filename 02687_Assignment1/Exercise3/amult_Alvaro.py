# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 17:01:07 2025

@author: alvar
"""

import numpy as np
from scipy.sparse import kron, eye, diags
import matplotlib.pyplot as plt
import scipy as sp
import time

def Amult(U,m):
    h = 1/(m+1)
    
    # Reshape U into a 2D array for easier indexing
    U = U.reshape((m, m), order='F')
    
    # Create expanded U to allow slicing
    expanded_U = np.zeros((m+2,m+2))
    expanded_U[1:-1,1:-1] = U
    
    # Initialize product array
    AU = np.zeros_like(U)
    
    # Stencil terms
    left_term   = expanded_U[1:-1, :-2]  # Shift left
    right_term  = expanded_U[1:-1, 2:]   # Shift right
    bottom_term = expanded_U[:-2, 1:-1]  # Shift down
    top_term    = expanded_U[2:, 1:-1]   # Shift up
    self_term   = -4*U                 # Center coefficient

    #Solution
    AU = (left_term+right_term+bottom_term+top_term+self_term)/h**2  
    
    return AU.flatten('F')
    
def poisson5(m):
    e = np.ones(m)
    S = diags([e, -2*e, e], [-1, 0, 1], shape=(m, m))
    I = eye(m)
    A = kron(I, S) + kron(S, I)
    A = (m + 1) ** 2 * A
    return A
        
m = 4     
U = np.ones(m**2)
# U = np.array(([1,1,1],[2,2,2],[3,3,3]))
U = U.flatten('F')
#U = [1,2,1,4,1,5,1,5,1]


# AMULT 
t1 = time.perf_counter()
AU = Amult(U, m)
AU_mult = AU.reshape((m, m), order='F')
t2 = time.perf_counter()
print(f'AMult Time is: {t2-t1}')

t1 = time.perf_counter()
A = poisson5(m).A
AU_poisson = A@U
AU_poisson = AU_poisson.reshape((m, m), order='F')
t2 = time.perf_counter()
print(f'Poisson Time is: {t2-t1}')