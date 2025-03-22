# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 16:33:16 2025

@author: alvar
"""

import numpy as np

def coarsen(r):
    m = int(np.sqrt(len(r)))
    r = r.reshape((m, m), order='F')
    m_coarse = int((m - 1) / 2)
    
    # Direct Interpolation
    # r_coarse = r[1::2,1::2]      # Pick each two elements from the first one
    
    # Full-Weighting Interpolation
    r_coarse = (
        1/16 * r[0:-2:2, 0:-2:2] + 1/8 * r[1:-1:2, 0:-2:2] + 1/16 * r[2::2, 0:-2:2] +
        1/8  * r[0:-2:2, 1:-1:2] + 1/4  * r[1:-1:2, 1:-1:2] + 1/8  * r[2::2, 1:-1:2] +
        1/16 * r[0:-2:2, 2::2] + 1/8 * r[1:-1:2, 2::2] + 1/16 * r[2::2, 2::2]
    )
    
    return r_coarse.flatten('F')

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





