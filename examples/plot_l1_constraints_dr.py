"""
===================================
Basis Pursuit with Douglas Rachford
===================================

Test DR for a standard constrained l1-minimization
"""
# Author: Samuel Vaiter <samuel.vaiter@gmail.com>
from __future__ import print_function, division
print(__doc__)

# modules
import time

import numpy as np
import scipy.linalg as lin
import matplotlib.pylab as plt

from pyprox import douglas_rachford
from pyprox.operators import soft_thresholding
from pyprox.context import Context

# Dimension of the problem
n = 500
p = n // 4

# Matrix and observations
A = np.random.randn(p, n)
y = np.random.randn(p, 1)

# operator callbacks
prox_f = soft_thresholding
prox_g = lambda x, tau: x + np.dot(A.T, lin.solve(np.dot(A, A.T),
    y - np.dot(A, x)))

# context
ctx = Context(full_output=True, maxiter=1000)
ctx.callback = lambda x: lin.norm(x, 1)

t1 = time.time()
x, fx = douglas_rachford(prox_f, prox_g, np.zeros((n, 1)), context=ctx)
t2 = time.time()
print("Performed 1000 iterations in " + str(t2 - t1) + " seconds.")

plt.plot(fx)
plt.show()
