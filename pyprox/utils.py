"""
Misc utils
"""
# Author: Samuel Vaiter <samuel.vaiter@ceremade.dauphine.fr>

from __future__ import division
import numpy as np

def soft_thresholding(x, gamma):
    return np.maximum(0, 1 - gamma / np.maximum(np.abs(x), 1E-10)) * x

def operator_norm(linop, n=None, maxiter=30, check=False):
    if hasattr(linop, 'norm') and not check:
        return linop.norm
    if n is None:
        n = np.random.randn(linop.dim[1], 1)
    if np.size(n) == 1:
        u = np.random.randn(n, 1)
    else:
        u = n
    u = u / np.linalg.norm(u)
    e = []
    for i in range(maxiter):
        #TODO check validity of this formula
        if hasattr(linop,'T'):
            v = linop.T(linop(u))
        else:
            # assume square (implicit) operator
            v = linop(u)
        e.append((u[:] * v[:]).sum())
        u = v / np.linalg.norm(v[:])
    L = e[-1]
    return L

def dual_prox(prox):
    return (lambda u,sigma: u - sigma*prox(u/sigma,1/sigma))