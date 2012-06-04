"""
Misc utils
"""
# Author: Samuel Vaiter <samuel.vaiter@ceremade.dauphine.fr>

from __future__ import division
import numpy as np

def operator_norm(linop, n=None, maxiter=30, check=False):
    if hasattr(linop, 'norm') and not check:
        return linop.norm
    if n is None:
        n = np.random.randn(linop.dim[1], 1)
    if np.size(n) == 1:
        u = np.random.randn(n, 1)
    else:
        u = n
    unorm = np.linalg.norm(u)
    if unorm > 1e-10:
        u = u / unorm
    else:
        return 0
    e = []
    for i in range(maxiter):
        if hasattr(linop,'T'):
            v = linop.T(linop(u))
        else:
            # assume square (implicit) operator
            v = linop(u)
        e.append((u[:] * v[:]).sum())
        vnorm = np.linalg.norm(v[:])
        if vnorm > 1e-10:
            u = v / vnorm
        else :
            return 0
    L = e[-1]
    return L