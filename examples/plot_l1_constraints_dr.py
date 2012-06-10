"""
===================================
Basis Pursuit with Douglas Rachford
===================================

"""
# Author: Samuel Vaiter <samuel.vaiter@ceremade.dauphine.fr>
from __future__ import division

print __doc__

# modules
import time

import numpy as np
import pylab as pl

from pyprox import douglas_rachford
from pyprox.operators import soft_thresholding

# Dimension of the problem
n = 500
p = n//4

# Matrix and observations
A = np.random.randn(p,n)
y = np.random.randn(p,1)

# operator callbacks
F = lambda x: np.linalg.norm(x,1)
prox_f = soft_thresholding
prox_g = lambda x,tau: x + np.dot(A.T, np.linalg.solve(np.dot(A,A.T),
    y - np.dot(A,x)))

t1 = time.time()
x, fx = douglas_rachford(prox_f, prox_g, np.zeros((n,1)),
    maxiter=1000, full_output=1, retall=0, callback=F)
t2 = time.time()
print "Performed 1000 iterations in " + str(t2-t1) + " seconds."

pl.plot(fx)
pl.show()