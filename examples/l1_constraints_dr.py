"""
===================================
Basis Pursuit with Douglas Rachford
===================================

"""

# modules
from __future__ import division
import numpy as np
import pylab as plt
from pyprox.algorithms import douglas_rachford
from pyprox.utils import soft_thresholding

# Dimension of the problem
n = 500
p = n//4

# Matrix and observations
A = np.random.randn(p,n)
y = np.random.randn(p,1)

# operator callbacks
F = lambda x: np.linalg.norm(x,1)
ProxF = soft_thresholding
ProxG = lambda x,tau: x + np.dot(A.T, np.linalg.solve(np.dot(A,A.T),
    y - np.dot(A,x)))

x, fx = douglas_rachford(ProxF, ProxG, np.zeros((n,1)),
    maxiter=1000, full_output=1, retall=0, callback=F)

plt.plot(fx)
plt.show()